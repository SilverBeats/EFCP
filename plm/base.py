import torch.nn as nn


class EncoderDecoder:

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
        encoder = self.encoder if not self.encoder_decoder_is_same else getattr(self, self.decoder_name)
        # input_ids.shape = (bs, len)
        outputs = encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            additive_embeds=additive_embeds,
            token_type_ids=token_type_ids,
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
        decoder = getattr(self, self.decoder_name)
        outputs = decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            additive_embeds=additive_embeds,
            token_type_ids=token_type_ids,
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
