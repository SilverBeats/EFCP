import abc
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from plm import GPT2EncoderDecoderModel, MyGPT2LMHeadModel


class BaseModel(nn.Module):
    def __init__(
            self,
            plm: Union[MyGPT2LMHeadModel, GPT2EncoderDecoderModel],
            tokenizer: GPT2Tokenizer
    ):
        super(BaseModel, self).__init__()
        self.plm = plm
        self.tokenizer = tokenizer
        self.config = plm.config
        self.hs = plm.config.n_embd

    def word_embedding(self, input_ids: Tensor) -> Tensor:
        if isinstance(self.plm, GPT2EncoderDecoderModel):
            return self.plm.encoder.get_input_embeddings()(input_ids)
        return self.plm.get_input_embeddings()(input_ids)

    def encode_sequence(
            self,
            input_ids: Tensor = None,
            inputs_embeds: Tensor = None,
            labels: Tensor = None,
            token_type_ids: Tensor = None,
            attention_mask: Tensor = None,
            position_ids: Tensor = None,
            additive_embeds: Tensor = None,
            past_key_values: Tuple[Tuple[Tensor]] = None,
            use_cache: bool = False,
            output_hidden_states: bool = True
    ) -> Tuple:
        if (input_ids is None and inputs_embeds is None) or (input_ids is not None and inputs_embeds is not None):
            raise ValueError

        pad_token_id = self.tokenizer.pad_token_id

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.ne(pad_token_id).float()

        if attention_mask is not None and position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1).long() - 1

        if additive_embeds is not None:
            dims = additive_embeds.dim()
            assert dims in [2, 3]
            if dims == 2:
                # (bs, 1, hs)
                additive_embeds = additive_embeds.unsqueeze(1)

        if isinstance(self.plm, GPT2EncoderDecoderModel):
            # (hidden_state, loss, p_k_v, attn)
            outputs = self.plm.encode(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                additive_embeds=additive_embeds,
                output_hidden_states=output_hidden_states,
            )
        else:
            # (lm_logits, p_k_v, hidden_state)
            outputs = self.plm(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                additive_embeds=additive_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )
        return outputs

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        kwargs.update({'use_cache': True})
        if isinstance(self.plm, GPT2EncoderDecoderModel):
            lm_logits, past_key_values, *_ = self.plm.decode(*args, **kwargs)
        else:
            lm_logits, past_key_values, *_ = self.plm(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values
        }

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError
