import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer

from plm import MyGPT2LMHeadModel
from utils.generate import generate
from .base import BaseModel


class Vanilla(BaseModel):
    def __init__(self, plm: MyGPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        super(Vanilla, self).__init__(plm, tokenizer)

    def forward(
            self,
            ctx_input_ids: Tensor,
            ctx_token_type_ids: Tensor,
            ctx_len: Tensor,
            tgt_input_ids: Tensor,
            tgt_label_ids: Tensor,
            tgt_token_type_ids: Tensor,
            do_eval: bool = False,
            do_gen: bool = False,
            **kwargs
    ):
        pad_token_id = self.tokenizer.pad_token_id
        ctx_p_k_v = self.encode_sequence(
            input_ids=ctx_input_ids,
            token_type_ids=ctx_token_type_ids,
            use_cache=True
        )[1]

        tgt_seq_len = tgt_input_ids.shape[1]
        tgt_attention_mask = torch.cat([
            ctx_input_ids.ne(pad_token_id).float(),
            tgt_input_ids.ne(pad_token_id).float()
        ], dim=-1)

        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1)[:, -tgt_seq_len:].long() - 1

        if do_gen:
            return tgt_attention_mask, tgt_position_ids, ctx_p_k_v

        lm_logits = self.encode_sequence(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            position_ids=tgt_position_ids,
            token_type_ids=tgt_token_type_ids,
            past_key_values=ctx_p_k_v
        )[0]

        loss = F.cross_entropy(
            input=lm_logits.view(-1, lm_logits.size(-1)),
            target=tgt_label_ids.view(-1),
            ignore_index=-1,
            reduction='none'
        ).view(tgt_label_ids.shape)
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(loss)
        loss_value = torch.sum(loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if do_eval:
            return loss, label_size
        return {
            'loss': loss_value,
            'ppl': ppl_value,
        }

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        (
            tgt_attention_mask, tgt_position_ids, ctx_p_k_v
        ) = self.forward(**forward_dict, do_gen=True)

        return {}, generate(self,
                            input_ids=forward_dict['tgt_input_ids'],
                            position_ids=tgt_position_ids,
                            attention_mask=tgt_attention_mask,
                            past_key_values=ctx_p_k_v,
                            **kwargs
                            )
