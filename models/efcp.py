from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Tokenizer
from transformers.modeling_utils import SequenceSummary

from plm import GPT2EncoderDecoderModel
from utils.generate import generate
from .base import BaseModel
from .modules import PredictEmpathyFactorsModule


class EFCP(BaseModel):
    def __init__(
            self,
            plm: GPT2EncoderDecoderModel,
            tokenizer: GPT2Tokenizer,
            model_config: Dict[str, Any]
    ):
        super(EFCP, self).__init__(plm, tokenizer)
        self.model_config = model_config
        self.pred_comae_coef = model_config['pred_comae_coef']
        self.decode_lm_coef = model_config['decode_lm_coef']

        if model_config['use_ef']:
            self.pef_module = PredictEmpathyFactorsModule(self.hs, 'elu')

        if model_config['use_ef']:
            self.ctx_summary_head = SequenceSummary(self.config)
            if model_config['predict_with_persona']:
                self.persona_summary_head = SequenceSummary(self.config)
                self.cp_linear = nn.Sequential(nn.Linear(2 * self.hs, self.hs), nn.ELU())

    def fuse_ctx_cs(
            self,
            ctx_h_s: Tensor,
            cs_h_s_dict: Dict[str, Tensor],
            cs_true_len: Dict[str, Tensor] = None
    ) -> Tensor:
        return self.acs(ctx_h_s, cs_h_s_dict, cs_true_len)

    def fuse_ctx_persona(
            self,
            ctx_h_s: Tensor,
            ctx_len: Tensor,
            p_h_s: Tensor = None,
            persona_len: Tensor = None,
    ) -> Tensor:
        ctx_last_token_hidden_states = self.ctx_summary_head(ctx_h_s, ctx_len - 1)

        if p_h_s is None:
            return ctx_last_token_hidden_states

        persona_last_token_hidden_states = self.persona_summary_head(p_h_s, persona_len - 1)
        t = self.cp_linear(torch.cat([ctx_last_token_hidden_states, persona_last_token_hidden_states], dim=-1))

        return t

    def forward_step(
            self,
            ctx_input_ids: Tensor,  # (bs, _len)
            ctx_token_type_ids: Tensor,
            ctx_len: Tensor,  # (bs, )
            ctx_da: Tensor,
            ctx_em: Tensor,
            ctx_cs_dict: Dict[str, Tensor],
            persona_input_ids: Tensor,
            persona_len: Tensor,  # (bs, )
            persona_token_type_ids: Tensor,
            tgt_input_ids: Tensor,
            tgt_token_type_ids: Tensor,
            tgt_da: Tensor,  # (bs, 1)
            tgt_em: Tensor,
            tgt_er: Tensor,
            tgt_ip: Tensor,
            tgt_ex: Tensor,
            **kwargs
    ):
        ctx_additive_embeds = None
        tgt_additive_embeds = None
        result_info = None

        if self.model_config['use_ef']:
            # (bs, 1, hs)
            ctx_additive_embeds = self.pef_module.embed_da(ctx_da) + \
                                  self.pef_module.embed_em(ctx_em)

        ctx_h_s = self.encode_sequence(
            input_ids=ctx_input_ids,
            additive_embeds=ctx_additive_embeds,
            token_type_ids=ctx_token_type_ids
        )[0]

        if self.model_config['use_persona']:
            p_h_s = self.encode_sequence(
                input_ids=persona_input_ids,
                token_type_ids=persona_token_type_ids
            )[0]

        if self.model_config['use_ef']:
            if self.model_config['predict_with_persona']:
                # (bs, hs)
                t = self.fuse_ctx_persona(ctx_h_s, ctx_len, p_h_s, persona_len)
            else:
                # (bs, hs)
                t = self.fuse_ctx_persona(ctx_h_s, ctx_len)
            result_info = self.pef_module(t, tgt_er, tgt_ip, tgt_ex, tgt_da, tgt_em)
            # (bs, hs)
            tgt_additive_embeds = result_info.pop('cm_embeds') + \
                                  result_info.pop('da_embeds') + \
                                  result_info.pop('em_embeds')
            # (bs, tgt_len, bs)
            tgt_additive_embeds = tgt_additive_embeds.unsqueeze(1).expand(-1, tgt_input_ids.shape[1], -1)

            result_info['pred_ef_loss'] = result_info.pop('loss')

        enc_contexts = [ctx_h_s]
        if self.model_config['use_persona']:
            enc_contexts.append(p_h_s)

        tgt_attention_mask = tgt_input_ids.ne(self.tokenizer.pad_token_id).float()
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1).long() - 1

        d = {
            'input_ids': tgt_input_ids,
            'additive_embeds': tgt_additive_embeds,
            'attention_mask': tgt_attention_mask,
            'position_ids': tgt_position_ids,
            'enc_contexts': enc_contexts,
            'token_type_ids': tgt_token_type_ids
        }
        return d, result_info

    def forward(
            self,
            ctx_input_ids: Tensor,  # (bs, _len)
            ctx_token_type_ids: Tensor,
            ctx_len: Tensor,  # (bs, )
            ctx_da: Tensor,
            ctx_em: Tensor,
            ctx_cs_dict: Dict[str, Tensor],
            persona_input_ids: Tensor,
            persona_len: Tensor,  # (bs, )
            persona_token_type_ids: Tensor,
            tgt_input_ids: Tensor,
            tgt_token_type_ids: Tensor,
            tgt_label_ids: Tensor,
            tgt_da: Tensor,  # (bs, 1)
            tgt_em: Tensor,
            tgt_er: Tensor,
            tgt_ip: Tensor,
            tgt_ex: Tensor,
            do_eval: bool = False,
            **kwargs
    ):
        d, result_info = self.forward_step(
            ctx_input_ids=ctx_input_ids,
            ctx_token_type_ids=ctx_token_type_ids,
            ctx_len=ctx_len,
            ctx_da=ctx_da,
            ctx_em=ctx_em,
            ctx_cs_dict=ctx_cs_dict,
            persona_input_ids=persona_input_ids,
            persona_len=persona_len,
            persona_token_type_ids=persona_token_type_ids,
            tgt_input_ids=tgt_input_ids,
            tgt_token_type_ids=tgt_token_type_ids,
            tgt_da=tgt_da,
            tgt_em=tgt_em,
            tgt_er=tgt_er,
            tgt_ip=tgt_ip,
            tgt_ex=tgt_ex
        )

        lm_logits = self.plm.decode(**d)[0]

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

        if do_eval:
            return lm_loss, label_size

        total_loss = lm_loss_value
        if self.pred_comae_coef != 0.0:
            total_loss = total_loss + self.pred_comae_coef * result_info.pop('pred_ef_loss')
        return {
            'loss': total_loss,
            'ppl': ppl_value
        }

    @torch.no_grad()
    def generate(self, forward_dict: Dict[str, Any], **kwargs):
        d, result_info = self.forward_step(**forward_dict)
        return result_info, generate(self, **d, **kwargs)
