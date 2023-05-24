import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer

from utils.generate import generate
from .base import BaseModel
from .gpt2 import GPT2EncoderDecoderModel


class MultiGPT2(BaseModel):
    def __init__(
            self,
            plm: GPT2EncoderDecoderModel,
            tokenizer: GPT2Tokenizer,
            **kwargs
    ):
        super(MultiGPT2, self).__init__(plm, tokenizer)

    def forward(
            self,
            ctx_input_ids: Tensor,
            ctx_token_type_ids: Tensor,
            tgt_input_ids: Tensor,
            tgt_label_ids: Tensor,
            tgt_token_type_ids: Tensor,
            persona_input_ids: Tensor,
            persona_token_type_ids: Tensor,
            do_eval: bool = False,
            do_gen: bool = False,
            **kwargs
    ):
        ctx_h_s, encode_ctx_loss, *_ = self.encode_sequence(
            input_ids=ctx_input_ids,
            labels=ctx_input_ids,
            token_type_ids=ctx_token_type_ids
        )

        p_h_s, encode_p_loss, *_ = self.encode_sequence(
            input_ids=persona_input_ids,
            labels=persona_input_ids,
            token_type_ids=persona_token_type_ids
        )

        enc_contexts = [ctx_h_s, p_h_s]
        if do_gen:
            return {
                'input_ids': tgt_input_ids,
                'enc_contexts': enc_contexts,
                'token_type_ids': tgt_token_type_ids
            }

        lm_logits = self.plm.decode(
            input_ids=tgt_input_ids,
            token_type_ids=tgt_token_type_ids,
            enc_contexts=enc_contexts
        )[0]

        lm_loss = F.cross_entropy(
            input=lm_logits.view(-1, lm_logits.size(-1)),
            target=tgt_label_ids.view(-1),
            ignore_index=-1,
            reduction='none'
        ).view(tgt_label_ids.shape)
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        if do_eval:
            return lm_loss, label_size

        total_loss = lm_loss_value + encode_ctx_loss + encode_p_loss

        return {
            'loss': total_loss,
            'ppl': ppl_value,
        }

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        d = self.forward(**forward_dict, do_gen=True)
        return {}, generate(self, **d, **kwargs)
