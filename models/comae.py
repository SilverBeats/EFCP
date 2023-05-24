import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer

from plm import MyGPT2LMHeadModel
from utils.generate import generate
from .base import BaseModel
from .modules import PredictEmpathyFactorsModule, gather_by_len


class CoMAE(BaseModel):
    def __init__(
            self,
            plm: MyGPT2LMHeadModel,
            tokenizer: GPT2Tokenizer,
            **kwargs
    ):
        super(CoMAE, self).__init__(plm, tokenizer)
        self.pef = PredictEmpathyFactorsModule(self.hs, 'tanh')

    def forward(
            self,
            ctx_input_ids: Tensor,  # (bs, ctx_seq_len)
            ctx_token_type_ids: Tensor,
            ctx_len: Tensor,  # (bs, )
            ctx_da: Tensor,  # (bs, 1)
            ctx_em: Tensor,
            tgt_input_ids: Tensor,  # (bs, tgt_seq_len)
            tgt_token_type_ids: Tensor,
            tgt_label_ids: Tensor,
            tgt_da: Tensor,  # (bs, 1)
            tgt_em: Tensor,
            tgt_er: Tensor,
            tgt_ip: Tensor,
            tgt_ex: Tensor,
            do_eval: bool = False,
            do_gen: bool = False,
            **kwargs
    ):
        pad_token_id = self.tokenizer.pad_token_id
        tgt_seq_len = tgt_input_ids.shape[1]

        ctx_additive_embeds = self.pef.embed_da(ctx_da) + self.pef.embed_em(ctx_em)
        # (bs, seq_len, hs)
        outputs = self.encode_sequence(
            input_ids=ctx_input_ids,
            token_type_ids=ctx_token_type_ids,
            additive_embeds=ctx_additive_embeds,
            use_cache=True
        )
        ctx_h_s = outputs[-1]
        ctx_p_k_v = outputs[1]
        # (bs, hs)
        last_token_h_s = gather_by_len(ctx_h_s, ctx_len)
        result_info = self.pef(last_token_h_s, tgt_er, tgt_ip, tgt_ex, tgt_da, tgt_em)
        # (bs, hs)
        tgt_additive_embeds = result_info.pop('cm_embeds') + \
                              result_info.pop('da_embeds') + \
                              result_info.pop('em_embeds')
        # (bs, ctx_seq_len + tgt_seq_len)
        tgt_attention_mask = torch.cat([
            ctx_input_ids.ne(pad_token_id).float(),
            tgt_input_ids.ne(pad_token_id).float()
        ], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1)[:, -tgt_seq_len:].long() - 1

        if do_gen:
            result_info.pop('loss')
            return (
                result_info,
                tgt_additive_embeds,
                tgt_attention_mask,
                tgt_position_ids,
                ctx_p_k_v
            )

        lm_logits = self.encode_sequence(
            input_ids=tgt_input_ids,
            token_type_ids=tgt_token_type_ids,
            attention_mask=tgt_attention_mask,
            position_ids=tgt_position_ids,
            additive_embeds=tgt_additive_embeds,
            past_key_values=ctx_p_k_v
        )[0]

        # (bs, tgt_len)
        lm_loss = F.cross_entropy(
            input=lm_logits.view(-1, lm_logits.size(-1)),
            target=tgt_label_ids.view(-1),
            ignore_index=-1,
            reduction='none'
        ).view(tgt_label_ids.shape)
        # (bs, )
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        predict_loss = result_info.pop('loss')

        if do_eval:
            return lm_loss, label_size

        total_loss = lm_loss_value + predict_loss
        return {
            'loss': total_loss,
            'ppl': ppl_value
        }

    def generate(self, forward_dict, **kwargs):
        (
            result_info,
            tgt_additive_embeds,
            tgt_attention_mask,
            tgt_position_ids,
            ctx_p_k_v
        ) = self.forward(**forward_dict, do_gen=True)

        return result_info, generate(self,
                                     input_ids=forward_dict['tgt_input_ids'],
                                     position_ids=tgt_position_ids,
                                     attention_mask=tgt_attention_mask,
                                     past_key_values=ctx_p_k_v,
                                     additive_embeds=tgt_additive_embeds,
                                     **kwargs
                                     )
