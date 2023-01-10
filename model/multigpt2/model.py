import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.gpt2_2 import GPT2EncoderDecoderModel
from utils.generate import generate


class MultiGPT2(nn.Module):
    def __init__(self, gpt2: GPT2EncoderDecoderModel, **kwargs):
        super(MultiGPT2, self).__init__()
        self.gpt2 = gpt2
        self.config = gpt2.config
        self.encode_lm_coef = kwargs.get('encode_lm_coef')

    def forward(self,
                ctx_input_ids, ctx_token_type_ids,
                tgt_input_ids, tgt_label_ids, tgt_token_type_ids,
                persona_input_ids, persona_token_type_ids,
                do_eval=False, do_gen=False, **kwargs):
        ctx_h_s, encode_ctx_loss = self.gpt2.encode(input_ids=ctx_input_ids,
                                                    labels=ctx_input_ids,
                                                    token_type_ids=ctx_token_type_ids)

        p_h_s, encode_p_loss = self.gpt2.encode(input_ids=persona_input_ids,
                                                labels=persona_input_ids,
                                                token_type_ids=persona_token_type_ids)

        enc_contexts = [ctx_h_s, p_h_s]
        if do_gen:
            return {
                'input_ids': tgt_input_ids,
                'enc_contexts': enc_contexts,
                'token_type_ids': tgt_token_type_ids
            }

        lm_logits = self.gpt2.decode(input_ids=tgt_input_ids,
                                     token_type_ids=tgt_token_type_ids,
                                     enc_contexts=enc_contexts)[0]

        lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                                  ignore_index=-1, reduction='none').view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        if do_eval:
            return lm_loss, label_size

        total_loss = lm_loss_value + self.encode_lm_coef * (encode_ctx_loss + encode_p_loss)

        return {
            'all': total_loss,
            'ppl': ppl_value,
            'ectx': encode_ctx_loss,
            'ep': encode_p_loss
        }

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        lm_logits, past_key_values, *_ = self.gpt2.decode(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values,
        }

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        d = self.forward(**forward_dict, do_gen=True)
        return {}, generate(self, **d, **kwargs)
