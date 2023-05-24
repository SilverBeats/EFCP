from typing import Any, Tuple
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer
from transformers.modeling_utils import SequenceSummary

from plm import MyGPT2LMHeadModel
from utils.generate import generate
from .base import BaseModel
from .modules import MLP
from .modules import gather_by_len
from .torchnlp import Encoder


class CEM(BaseModel):
    def __init__(self, plm: MyGPT2LMHeadModel, tokenizer: GPT2Tokenizer, model_config: Dict[str, Any]):
        super(CEM, self).__init__(plm, tokenizer)
        encoder_config = model_config.get('encoder_config')
        self.cog_refined_encoder = Encoder(2 * self.hs, self.hs, **encoder_config)
        self.emo_refined_encoder = Encoder(2 * self.hs, self.hs, **encoder_config)
        self.total_refined_encoder = MLP(5 * self.hs, 3 * self.hs, self.hs, 'relu')
        # to predict the emotion id
        self.cem_choice_head = SequenceSummary(self.config)
        self.emo_lin = nn.Linear(self.hs, 10, bias=False)

    def forward(
            self,
            ctx_input_ids: Tensor,
            ctx_token_type_ids: Tensor,
            ctx_em: Tensor,
            ctx_len: Tensor,
            ctx_cs_dict: Dict[str, Tensor],
            tgt_input_ids: Tensor,
            tgt_label_ids: Tensor,
            tgt_token_type_ids: Tensor,
            do_eval: bool = False,
            do_gen: bool = False,
            **kwargs
    ):
        pad_token_id = self.tokenizer.pad_token_id
        # (bs, ctx_len, hs)
        ctx_h_s = self.encode_sequence(ctx_input_ids, token_type_ids=ctx_token_type_ids)[-1]
        ctx_cs_token_type_ids = ctx_token_type_ids[:, :1]
        cs_h_s_dict = self.encode_cs_dict(ctx_cs_dict, ctx_cs_token_type_ids)
        ctx_h_s, pred_em_loss = self.fuse_ctx_cs(ctx_h_s, cs_h_s_dict, ctx_len, ctx_em)

        p_k_v = self.encode_sequence(
            inputs_embeds=ctx_h_s,
            token_type_ids=ctx_token_type_ids,
            use_cache=True
        )[1]

        tgt_attention_mask = torch.cat([
            ctx_input_ids.ne(pad_token_id).float(),
            tgt_input_ids.ne(pad_token_id).float()
        ], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1).long()[:, -tgt_input_ids.shape[1]:] - 1

        if do_gen:
            d = {
                'input_ids': tgt_input_ids,
                'attention_mask': tgt_attention_mask,
                'position_ids': tgt_position_ids,
                'token_type_ids': tgt_token_type_ids,
                'past_key_values': p_k_v
            }
            return d

        lm_logits = self.encode_sequence(
            input_ids=tgt_input_ids,
            token_type_ids=tgt_token_type_ids,
            position_ids=tgt_position_ids,
            attention_mask=tgt_attention_mask,
            past_key_values=p_k_v
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

        total_loss = lm_loss_value + pred_em_loss

        return {'loss': total_loss, 'ppl': ppl_value}

    def encode_cs_dict(
            self,
            ctx_cs_dict: Dict[str, Tensor],
            ctx_cs_token_ids: Tensor
    ) -> Dict[str, Tensor]:
        ctx_cs_token_ids = ctx_cs_token_ids.view(-1, 1)
        pad_token_id = self.tokenizer.pad_token_id
        device = ctx_cs_token_ids.device
        idxes = torch.arange(0, ctx_cs_token_ids.shape[0]).to(device)

        cs_h_s_dict = dict()
        for k, v in ctx_cs_dict.items():
            # (bs, )
            true_len = v.ne(pad_token_id).sum(keepdims=True, dim=1).long()
            token_type_ids = ctx_cs_token_ids.expand(-1, v.shape[1])
            # (bs, seq_len, hs)
            value_h_s = self.encode_sequence(v, token_type_ids=token_type_ids)[-1]
            if 'xReact' in k or 'xAttr' in k:
                # (bs, hs)
                value_h_s = torch.stack([torch.mean(value_h_s[i, :l, :], dim=0) for i, l in zip(idxes, true_len)])
            else:
                # (bs, hs)
                value_h_s = gather_by_len(value_h_s, true_len)
            cs_h_s_dict[k] = value_h_s

        return cs_h_s_dict

    def fuse_ctx_cs(
            self,
            ctx_h_s: Tensor,
            cs_h_s_dict: Dict[str, Tensor],
            ctx_len: Tensor,
            ctx_em: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ctx_seq_len = ctx_h_s.shape[1]
        refined_h_s = []  # [(bs, ctx_seq_len, hs)] * 5
        pred_em_loss = None
        for k, v in cs_h_s_dict.items():
            # (bs, ctx_seq_len, hs)
            v = v.unsqueeze(1).expand(-1, ctx_seq_len, -1)
            if 'xReact' in k:
                vv = self.emo_refined_encoder(torch.cat([ctx_h_s, v], dim=-1))
                ctx_em_logits = self.emo_lin(self.cem_choice_head(vv, ctx_len - 1))
                pred_em_loss = F.cross_entropy(ctx_em_logits, ctx_em.view(-1), reduction='mean')
            else:
                vv = self.cog_refined_encoder(torch.cat([ctx_h_s, v], dim=-1))
            refined_h_s.append(vv)
        refined_h_s = torch.cat(refined_h_s, dim=-1)  # (bs, ctx_seq_len, 5 * hs)
        ctx_h_s = self.total_refined_encoder(nn.Sigmoid()(refined_h_s) * refined_h_s)

        return ctx_h_s, pred_em_loss

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        d = self.forward(**forward_dict, do_gen=True)
        return {}, generate(self, **d, **kwargs)
