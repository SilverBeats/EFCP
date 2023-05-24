from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Conv1D, GPT2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2MLP
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_conv1d_layer


class Attention(nn.Module):
    def __init__(self, config, fuse_attention=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions
        self.scale_attn_weights = config.scale_attn_weights
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head

        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        n_ctx = config.n_ctx
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self.n_head = config.n_head
        self.split_size = self.embed_dim

        if not fuse_attention:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, bias_mask=True):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        if bias_mask:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights += attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.n_head * self.head_dim,)
        return x.view(new_x_shape)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _split_heads(self, tensor, k=False):
        """
        Same to GPT2Attention's _split_heads
        Splits hidden_size dim into head_dim and n_head
        tensor.shape = (bs, len, n_embd=768)
        """
        new_shape = tensor.size()[:-1] + (self.n_head, self.head_dim)
        tensor = tensor.view(new_shape)
        # (bs, n_head, len, head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self,
                x,
                k=None,
                attention_mask=None,
                layer_past=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
                ):
        r"""
        When k is None, it is still self-attention for input x,
        while k is not None, it becomes the mutual attention
        between x and k in which k is used to generate key and value
        """
        if k is None:
            query, key, value = self.c_attn(x).split(self.split_size, dim=2)
            query = self._split_heads(query)
            key = self._split_heads(key)
            value = self._split_heads(value)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
        else:
            proj_weight, proj_bias = self.c_attn.weight, self.c_attn.bias
            size_out = x.size()[:-1] + (self.split_size,)
            query = torch.addmm(proj_bias[: self.split_size], x.view(-1, x.size(-1)),
                                proj_weight[:, :self.split_size])
            query = self._split_heads(query.view(*size_out))
            if layer_past is None:
                size_out = k.size()[:-1] + (self.split_size * 2,)
                kv = torch.addmm(proj_bias[self.split_size:], k.view(-1, k.size(-1)),
                                 proj_weight[:, self.split_size:])
                kv = kv.view(size_out)
                key, value = kv.split(self.split_size, dim=2)
                key = self._split_heads(key)
                value = self._split_heads(value)
            else:
                key, value = layer_past

        # (bs, n_head, query_len, head_dim); (bs, n_head, query_len, key_len)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, bias_mask=(k is None))

        attn_output = self._merge_heads(attn_output)  # (bs, query_len, hs)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, query)
        if use_cache:
            present = (key, value)
            outputs = outputs + (present,)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config
        self.hs = hs = config.n_embd
        self.ln_1 = nn.LayerNorm(hs, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(hs, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4 * hs, config)

        self.output_attentions = config.output_attentions

        self.dropout = nn.Dropout(config.attn_pdrop)
        self.context_attns = None
        self.attention_module = None
        self.attention_fusion_type = None

    def attention_pooling(self, attention_list, layer_past=None):
        if self.attention_fusion_type == 'sw':
            return torch.mean(
                torch.stack(attention_list, dim=0) * self.attention_module.unsqueeze(-1).unsqueeze(-1),
                dim=0
            )
        elif self.attention_fusion_type == 'linear':
            return self.attention_module(torch.cat(attention_list, dim=-1))
        else:
            raise NotImplementedError('More detail in https://github.com/caoyu-noob/Multi-GPT2')

    def set_attention_pooling_module(self, attention_fusion_type, source_types):
        self.attention_fusion_type = attention_fusion_type
        if attention_fusion_type == 'sw':
            self.attention_module = torch.nn.Parameter(torch.ones(source_types + 1, 1) / (source_types + 1))
        elif attention_fusion_type == 'linear':
            self.attention_module = nn.Linear(self.hs * (source_types + 1), self.hs)
        else:
            raise NotImplementedError('More detail in https://github.com/caoyu-noob/Multi-GPT2')

    def set_context_attns(self, source_types):
        if source_types <= 0:
            return
        self.context_attns = nn.ModuleList([Attention(self.config) for _ in range(source_types)])

    def forward(self,
                hidden_states,
                attention_mask=None,
                encoded_context=None,
                layer_past=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
                ):
        if encoded_context is None:
            encoded_context = []

        attentions = () if output_attentions else None
        presents = () if use_cache else None

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past[0] if layer_past else None,
            head_mask=head_mask,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # self-attention result
        if use_cache:
            presents += (attn_outputs[2],)  # self-attention (key, value)
        if output_attentions:
            attentions += (attn_outputs[3],)  # self-attention weights

        if len(encoded_context) != 0:
            """f encoded persona and history is used as input"""
            context_attn_outputs = ()  # for attn-fuse
            present_context_k_v = () if use_cache else None

            for i, enc in enumerate(encoded_context):
                cur_layer_past = None if layer_past is None else layer_past[1][i]

                enc_attn_outputs = self.context_attns[i](hidden_states,
                                                         k=encoded_context[i],
                                                         layer_past=cur_layer_past,
                                                         use_cache=use_cache,
                                                         output_attentions=output_attentions,
                                                         )
                context_attn_outputs += (enc_attn_outputs[0],)
                if use_cache:
                    present_context_k_v += (enc_attn_outputs[2],)
                if output_attentions:
                    attentions += (enc_attn_outputs[3],)

            attn_pool_outputs = self.attention_pooling((attn_output,) + context_attn_outputs)

            attn_output = self.dropout(attn_pool_outputs)

            if use_cache:
                presents += (present_context_k_v,)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (presents,)
        if output_attentions:
            outputs += (attentions,)

        return outputs


class MyGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super(MyGPT2Model, self).__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
            self,
            enc_contexts: Optional[List] = None,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            additive_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ):
        if enc_contexts is None:
            enc_contexts = []

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states += token_type_embeds

        if additive_embeds is not None:
            hidden_states += additive_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                encoded_context=enc_contexts,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_attentions = all_attentions + (outputs[-1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class GPT2EncoderDecoderModel(GPT2LMHeadModel):

    def __init__(self, config):
        super(GPT2EncoderDecoderModel, self).__init__(config)
        self.transformer = MyGPT2Model(config)
        self.encoder = None
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        raise NotImplementedError

    def encode(
            self,
            input_ids=None,
            inputs_embeds=None,
            labels=None,
            additive_embeds=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        # input_ids.shape = (bs, len)
        outputs = self.encoder(input_ids=input_ids,
                               inputs_embeds=inputs_embeds,
                               additive_embeds=additive_embeds,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               past_key_values=past_key_values,
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states
                               )
        encode_output = (outputs.last_hidden_state,)

        if labels is not None:
            # (bs, len, vocab_size)
            lm_logits = self.lm_head(outputs.last_hidden_state)
            # (bs, len - 1, vocab_size)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # (bs, len - 1, vocab_size)
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            encode_output += (loss,)

        if use_cache:
            encode_output += (outputs.past_key_values,)
        if output_attentions:
            encode_output += (outputs.attentions,)
        return encode_output

    def decode(
            self,
            input_ids=None,
            inputs_embeds=None,
            additive_embeds=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            enc_contexts=None,
            **kwargs
    ):
        if enc_contexts is None:
            enc_contexts = []
        outputs = self.transformer(input_ids=input_ids,
                                   inputs_embeds=inputs_embeds,
                                   additive_embeds=additive_embeds,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   past_key_values=past_key_values,
                                   use_cache=use_cache,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   enc_contexts=enc_contexts,
                                   )
        last_hidden_state = outputs.last_hidden_state
        lm_logits = self.lm_head(last_hidden_state)

        decode_output = (lm_logits,)
        if use_cache:
            decode_output += (outputs.past_key_values,)
        if output_attentions:
            decode_output += (outputs.attentions,)
        return decode_output
