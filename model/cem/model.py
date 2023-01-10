import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import SequenceSummary

from model.module import MLP
from model.torchnlp import Encoder
from modeling import MyGPT2LMHeadModel
from utils.generate import generate


class CEM(nn.Module):
    def __init__(self, gpt2: MyGPT2LMHeadModel, token_id_dict, **kwargs):
        super(CEM, self).__init__()
        self.token_id_dict = token_id_dict
        self.gpt2 = gpt2
        self.config = gpt2.config
        self.hs = hs = gpt2.config.n_embd

        encoder_config = kwargs.get('encoder_config')
        self.cog_refined_encoder = Encoder(2 * hs, hs, **encoder_config)
        self.emo_refined_encoder = Encoder(2 * hs, hs, **encoder_config)
        self.total_refined_encoder = MLP(5 * hs, 3 * hs, hs)
        # to predict the emotion id
        self.cem_choice_head = SequenceSummary(self.config)
        self.emo_lin = nn.Linear(hs, 10, bias=False)

    def forward(self,
                ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em, ctx_len,
                ctx_xReact, ctx_xEffect, ctx_xWant, ctx_xIntent, ctx_xNeed,
                tgt_input_ids, tgt_label_ids, tgt_token_type_ids,
                do_eval=False, do_gen=False, **kwargs):
        result_info = dict()
        bs = ctx_input_ids.shape[0]
        idxs = torch.arange(0, bs).type_as(ctx_input_ids)

        pad = self.token_id_dict['pad']

        # (bs, ctx_len, hs)
        (
            ctx_h_s,
            ctx_attention_mask,
            ctx_position_ids
        ) = self.encode_ctx(ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em)

        cs_dict = {
            'ctx_xIntent': ctx_xIntent,
            'ctx_xWant': ctx_xWant,
            'ctx_xNeed': ctx_xNeed,
            'ctx_xReact': ctx_xReact,
            'ctx_xEffect': ctx_xEffect
        }
        ctx_h_s = self.acquire_common_sense(ctx_h_s, ctx_token_type_ids, cs_dict, ctx_em, ctx_len, result_info)

        p_k_v = self.gpt2(inputs_embeds=ctx_h_s,
                          token_type_ids=ctx_token_type_ids,
                          position_ids=ctx_position_ids,
                          attention_mask=ctx_attention_mask,
                          use_cache=True)[1]

        tgt_attention_mask = torch.cat([ctx_attention_mask, tgt_input_ids.ne(pad).float()], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1).long()[:, -tgt_input_ids.shape[1]:] - 1

        if do_gen:
            d = {
                'input_ids': tgt_input_ids,
                'attention_mask': tgt_attention_mask,
                'position_ids': tgt_position_ids,
                'token_type_ids': tgt_token_type_ids,
                'past_key_values': p_k_v
            }
            return d, result_info

        lm_logits = self.gpt2(input_ids=tgt_input_ids,
                              token_type_ids=tgt_token_type_ids,
                              position_ids=tgt_position_ids,
                              attention_mask=tgt_attention_mask,
                              past_key_values=p_k_v)[0]

        lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                                  ignore_index=-1, reduction='none').view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        if do_eval:
            return lm_loss, label_size

        em_loss = result_info.pop('ctx_em_loss')
        total_loss = lm_loss_value + em_loss

        return {'all': total_loss, 'em': em_loss, 'ppl': ppl_value}

    def encode_ctx(self, ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em):
        pad = self.token_id_dict['pad']
        ctx_attention_mask = ctx_input_ids.ne(pad).float()
        ctx_position_ids = torch.cumsum(ctx_attention_mask, dim=-1).long() - 1
        ctx_h_s = self.gpt2(input_ids=ctx_input_ids,
                            token_type_ids=ctx_token_type_ids,
                            position_ids=ctx_position_ids,
                            attention_mask=ctx_attention_mask)[-1]
        return ctx_h_s, ctx_attention_mask, ctx_position_ids

    def acquire_common_sense(self, ctx_h_s, ctx_token_type_ids, ctx_cs_dict, ctx_em, ctx_len, result_info):
        token_type_id = ctx_token_type_ids[:, 0].unsqueeze(1)  # (bs, 1)
        idxs = torch.arange(0, ctx_h_s.shape[0], device=ctx_h_s.device).long()
        pad = self.token_id_dict['pad']

        result = dict()
        # key in {ctx_React, ctx_Intent, ctx_Want, ctx_Need, ctx_Effect}
        for key, value in ctx_cs_dict.items():
            attention_mask = value.ne(pad).float()  # (bs, len)
            position_ids = torch.cumsum(attention_mask, dim=-1).long() - 1
            token_type_ids = token_type_id.repeat(1, value.shape[1])  # (bs, len)
            true_len = attention_mask.sum(-1).long()  # bs. exclude pad
            # (bs, len, hs)
            value_h_s = self.gpt2(input_ids=value,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids)[-1]
            if key.lower().find('react') != -1:
                # (bs, hs)
                value_h_s = torch.stack([torch.mean(value_h_s[i, :l, :], dim=0) for i, l in zip(idxs, true_len)])
                # (bs, ctx_len, hs)
                value_h_s = value_h_s.unsqueeze(1).repeat(1, ctx_h_s.shape[1], 1)
            else:
                # (bs, ctx_len, hs)
                value_h_s = value_h_s[idxs, true_len - 1].unsqueeze(1).repeat(1, ctx_h_s.shape[1], 1)
            result[key] = value_h_s

        for key, value in result.items():
            # (bs, ctx_len, 2 * hs)
            _input = torch.cat([ctx_h_s, value], dim=-1)
            if key.lower().find('react') != -1:
                # (bs, ctx_len, hs)
                result[key] = self.emo_refined_encoder(_input)
                # here, cem means context's emotion
                ctx_em_logits = self.emo_lin(self.cem_choice_head(result[key], ctx_len - 1))
                pred_em_top1 = torch.topk(ctx_em_logits, k=1, dim=-1)[1]
                pred_em_top3 = torch.topk(ctx_em_logits, k=3, dim=-1)[1]
                # (bs, 10)
                pred_ctx_em_loss = F.cross_entropy(ctx_em_logits, ctx_em.view(-1), reduction='mean')
                result_info.update({
                    'ctx_em_loss': pred_ctx_em_loss,
                    'pred_em': ctx_em.view(-1),
                    'pred_em_top1': pred_em_top1,
                    'pred_em_top3': pred_em_top3,
                })
            else:
                # (bs, ctx_len, hs)
                result[key] = self.cog_refined_encoder(_input)

        # (bs, 2 + ctx_len, 5 * hs)
        cem_refined = torch.cat([result[k] for k in result], dim=-1)
        # (bs, 2 + ctx_len, hs)
        ctx_h_s = self.total_refined_encoder(nn.Sigmoid()(cem_refined) * cem_refined)
        return ctx_h_s

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        lm_logits, past_key_values, *_ = self.gpt2(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values,
        }

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        d, result_info = self.forward(**forward_dict, do_gen=True)
        return result_info, generate(self, **d, **kwargs)
