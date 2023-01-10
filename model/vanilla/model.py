import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.generate import generate


class Vanilla(nn.Module):
    def __init__(self, gpt2, token_id_dict, **kwargs):
        super(Vanilla, self).__init__()
        self.gpt2 = gpt2
        self.config = gpt2.config
        self.hs = gpt2.config.n_embd
        self.token_id_dict = token_id_dict

    def forward(self, ctx_input_ids, ctx_token_type_ids, ctx_len, tgt_input_ids, tgt_label_ids, tgt_token_type_ids,
                tgt_len, do_eval=False, do_gen=False, **kwargs):
        result_info = dict()
        pad = self.token_id_dict['pad']
        bs = ctx_input_ids.shape[0]
        idxs = torch.arange(0, bs).type_as(ctx_input_ids)
        src_atten_mask = ctx_input_ids.ne(pad).float()
        _, ctx_p_k_v, ctx_h_s = self.gpt2(input_ids=ctx_input_ids,
                                          token_type_ids=ctx_token_type_ids,
                                          attention_mask=src_atten_mask,
                                          use_cache=True)
        # (bs, hs)
        ctx_h_s = ctx_h_s[idxs, ctx_len - 1]
        tgt_atten_mask = torch.cat([src_atten_mask,
                                    src_atten_mask.new_ones(
                                        (src_atten_mask.size(0), tgt_input_ids.size(1)))], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_atten_mask, dim=-1)[:, -tgt_input_ids.size(1):].type_as(ctx_input_ids) - 1

        if do_gen:
            return result_info, tgt_atten_mask, tgt_position_ids, ctx_p_k_v

        outputs = self.gpt2(
            input_ids=tgt_input_ids,
            position_ids=tgt_position_ids,
            attention_mask=tgt_atten_mask,  # should be concat with src_attention_mask
            token_type_ids=tgt_token_type_ids,
            past_key_values=ctx_p_k_v
        )
        lm_logits = outputs[0]

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                               ignore_index=-1, reduction='none')
        loss = loss.view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(loss)
        loss_value = torch.sum(loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if do_eval:
            return loss, label_size
        return {
            'all': loss_value,
            'ppl': ppl_value,
        }

    def generate(self, forward_dict, **kwargs):

        result_info, tgt_atten_mask, tgt_position_ids, ctx_p_k_v = self.forward(**forward_dict, do_gen=True)

        return result_info, generate(self,
                                     input_ids=forward_dict['tgt_input_ids'],
                                     position_ids=tgt_position_ids,
                                     attention_mask=tgt_atten_mask,
                                     past_key_values=ctx_p_k_v,
                                     **kwargs
                                     )

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        lm_logits, past_key_values, _ = self.gpt2(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values,
        }
