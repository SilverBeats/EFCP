import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.generate import generate


class CoMAE(nn.Module):
    def __init__(self, gpt2, token_id_dict, **kwargs):
        super(CoMAE, self).__init__()
        self.gpt2 = gpt2
        self.config = gpt2.config
        self.hs = hs = gpt2.config.n_embd
        self.token_id_dict = token_id_dict
        self.pred_comae_coef = kwargs.get('pred_comae_coef')

        self.er_embeddings = nn.Embedding(2, hs)
        self.ex_embeddings = nn.Embedding(2, hs)
        self.in_embeddings = nn.Embedding(2, hs)
        self.dialact_embeddings = nn.Embedding(9, hs)
        self.emotion_embeddings = nn.Embedding(10, hs)

        mean = 0.0
        std = gpt2.config.initializer_range
        self.er_embeddings.weight.data.normal_(mean, std)
        self.ex_embeddings.weight.data.normal_(mean, std)
        self.in_embeddings.weight.data.normal_(mean, std)
        self.dialact_embeddings.weight.data.normal_(mean, std)
        self.emotion_embeddings.weight.data.normal_(mean, std)

        self.er_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.ex_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.in_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.da_head = nn.Sequential(nn.Linear(2 * hs, hs), nn.ELU())
        self.em_head = nn.Sequential(nn.Linear(3 * hs, hs), nn.ELU())

    def forward(self, ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em, ctx_len, tgt_input_ids,
                tgt_label_ids, tgt_token_type_ids, tgt_da, tgt_em, tgt_er, tgt_ex, tgt_in, tgt_len,
                do_eval=False, do_gen=False, **kwargs):
        result_info = dict()
        pad = self.token_id_dict['pad']
        bs = ctx_input_ids.shape[0]
        idxs = torch.arange(0, bs).type_as(ctx_input_ids)
        ctx_attention_mask = ctx_input_ids.ne(pad).float()
        ctx_additive_embeds = self.dialact_embeddings(ctx_da) + \
                              self.emotion_embeddings(ctx_em)

        _, ctx_p_k_v, ctx_h_s = self.gpt2(input_ids=ctx_input_ids,
                                          token_type_ids=ctx_token_type_ids,
                                          attention_mask=ctx_attention_mask,
                                          additive_embeds=ctx_additive_embeds,
                                          use_cache=True)
        # (bs, hs)
        ctx_l_h_s = ctx_h_s[idxs, ctx_len - 1]
        self.pred_cm(ctx_l_h_s, tgt_er, tgt_ex, tgt_in, result_info)
        self.pred_da(ctx_l_h_s, tgt_da, result_info)
        self.pred_em(ctx_l_h_s, tgt_em, result_info)
        # (bs, hs)
        tgt_additive_embed = result_info.pop('cm_embeds') + \
                             result_info.pop('da_embeds') + \
                             result_info.pop('em_embeds')
        # (bs, tgt_len, bs)
        tgt_additive_embed = tgt_additive_embed.unsqueeze(1).expand(-1, tgt_input_ids.shape[1], -1)
        tgt_attention_mask = torch.cat([ctx_attention_mask,
                                        ctx_attention_mask.new_ones(
                                            (ctx_attention_mask.size(0), tgt_input_ids.size(1)))], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1)[:, -tgt_input_ids.size(1):].long() - 1

        if do_gen:
            return result_info, tgt_additive_embed, tgt_attention_mask, tgt_position_ids, ctx_p_k_v

        outputs = self.gpt2(
            input_ids=tgt_input_ids,
            position_ids=tgt_position_ids,
            attention_mask=tgt_attention_mask,  # should be concat with ctx_attention_mask
            token_type_ids=tgt_token_type_ids,
            additive_embeds=tgt_additive_embed,
            past_key_values=ctx_p_k_v
        )
        lm_logits = outputs[0]

        cm_loss = result_info['cm_loss']
        da_loss = result_info['da_loss']
        em_loss = result_info['em_loss']

        lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                                  ignore_index=-1, reduction='none').view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        all_loss = lm_loss_value + self.pred_comae_coef * (cm_loss + da_loss + em_loss)
        if not do_eval:
            res = {
                'all': all_loss,
                'ppl': ppl_value,
                'cm': cm_loss,
                'da': da_loss,
                'em': em_loss,
            }
            return res
        else:
            return lm_loss, label_size

    def pred_cm(self, ctx_l_h_s, tgt_er, tgt_ex, tgt_in, result_info):
        tgt_er_logits = F.linear(self.er_head(ctx_l_h_s), self.er_embeddings.weight)
        tgt_ex_logits = F.linear(self.ex_head(ctx_l_h_s), self.ex_embeddings.weight)
        tgt_in_logits = F.linear(self.in_head(ctx_l_h_s), self.in_embeddings.weight)
        # calc loss
        pred_er_loss = F.cross_entropy(tgt_er_logits, tgt_er.view(-1), reduction='mean')  # scalar
        pred_ex_loss = F.cross_entropy(tgt_ex_logits, tgt_ex.view(-1), reduction='mean')  # scalar
        pred_in_loss = F.cross_entropy(tgt_in_logits, tgt_in.view(-1), reduction='mean')  # scalar

        # get the pred result
        pred_er_top1 = torch.topk(tgt_er_logits, k=1, dim=-1)[1]
        pred_ex_top1 = torch.topk(tgt_ex_logits, k=1, dim=-1)[1]
        pred_in_top1 = torch.topk(tgt_in_logits, k=1, dim=-1)[1]
        # (bs, hs)
        cm_embeds = (self.er_embeddings(tgt_er) + self.ex_embeddings(tgt_ex) + self.in_embeddings(tgt_in)).squeeze(1)
        cm_loss = pred_er_loss + pred_ex_loss + pred_in_loss

        result_info.update({
            'pred_er': tgt_er,
            'pred_ex': tgt_ex,
            'pred_in': tgt_in,
            'pred_er_top1': pred_er_top1,
            'pred_in_top1': pred_in_top1,
            'pred_ex_top1': pred_ex_top1,
            'cm_embeds': cm_embeds,
            'cm_loss': cm_loss
        })

    def pred_da(self, ctx_l_h_s, tgt_da, result_info):
        tgt_da_logits = F.linear(
            self.da_head(torch.cat([ctx_l_h_s, result_info.get('cm_embeds')], dim=-1)),
            self.dialact_embeddings.weight
        )
        pred_da_loss = F.cross_entropy(tgt_da_logits, tgt_da.view(-1), reduction='mean')  # scalar
        pred_da_top1 = torch.topk(tgt_da_logits, k=1, dim=-1)[1]
        pred_da_top3 = torch.topk(tgt_da_logits, k=3, dim=-1)[1]
        # (bs, hs)
        da_embeds = self.dialact_embeddings(tgt_da.view(-1))
        result_info.update({
            'pred_da': tgt_da,
            'pred_da_top1': pred_da_top1,
            'pred_da_top3': pred_da_top3,
            'da_loss': pred_da_loss,
            'da_embeds': da_embeds
        })

    def pred_em(self, ctx_h_s, tgt_em, result_info):
        tgt_em_logits = F.linear(
            self.em_head(torch.cat([ctx_h_s, result_info.get('cm_embeds'), result_info.get('da_embeds')], dim=-1)),
            self.emotion_embeddings.weight)
        pred_em_loss = F.cross_entropy(tgt_em_logits, tgt_em.view(-1), reduction='mean')  # scalar
        pred_em_top1 = torch.topk(tgt_em_logits, k=1, dim=-1)[1]
        pred_em_top3 = torch.topk(tgt_em_logits, k=3, dim=-1)[1]
        # (bs, hs)
        em_embeds = self.emotion_embeddings(tgt_em.view(-1))
        result_info.update({
            'pred_em': tgt_em,
            'pred_em_top1': pred_em_top1,
            'pred_em_top3': pred_em_top3,
            'em_loss': pred_em_loss,
            'em_embeds': em_embeds
        })

    def generate(self, forward_dict, **kwargs):
        (
            result_info,
            tgt_additive_embed,
            tgt_atten_mask,
            tgt_position_ids,
            ctx_p_k_v
        ) = self.forward(**forward_dict, do_gen=True)

        return result_info, generate(self,
                                     input_ids=forward_dict['tgt_input_ids'],
                                     position_ids=tgt_position_ids,
                                     attention_mask=tgt_atten_mask,
                                     past_key_values=ctx_p_k_v,
                                     additive_embeds=tgt_additive_embed,
                                     **kwargs
                                     )

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        lm_logits, past_key_values, _ = self.gpt2(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values,
        }
