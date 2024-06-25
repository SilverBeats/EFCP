from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import top_k_top_p_filtering


@torch.no_grad()
def generate(
        self,
        input_ids=None,
        inputs_embeds=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        unk_token_id=None,
        cls_token_id=None,
        sep_token_id=None,
        usr_token_id=None,
        sys_token_id=None,
        persona_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        **model_kwargs
) -> torch.LongTensor:
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    unk_token_id = unk_token_id if unk_token_id is not None else self.config.unk_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    cls_token_id = cls_token_id if cls_token_id is not None else self.config.cls_token_id
    sep_token_id = sep_token_id if sep_token_id is not None else self.config.sep_token_id
    usr_token_id = usr_token_id if usr_token_id is not None else self.config.usr_token_id
    sys_token_id = sys_token_id if sys_token_id is not None else self.config.sys_token_id
    persona_token_id = persona_token_id if persona_token_id is not None else self.config.persona_token_id

    batch_size = input_ids.size(0)
    device = input_ids.device

    if not do_sample:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
    else:
        if inputs_embeds is None:
            assert position_ids.shape == input_ids.shape

    if token_type_ids is not None:
        assert token_type_ids.dim() == 2
        if inputs_embeds is None:
            assert token_type_ids.shape == input_ids.shape

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if num_return_sequences > 1 or num_beams > 1:
        def reshape(x):
            """
            the shape of x change to (effective_batch_size * num_beams, *x.shape[1:])
            """
            x_shape = x.size()
            x = x.unsqueeze(1).repeat(1, effective_batch_mult * num_beams, *((1,) * (len(x_shape) - 1)))
            x = x.contiguous().view(-1, *(x_shape[1:]))
            return x

        input_ids = reshape(input_ids)
        position_ids = reshape(position_ids)
        attention_mask = reshape(attention_mask)
        if token_type_ids is not None:
            token_type_ids = reshape(token_type_ids)

    if num_beams > 1:
        raise NotImplementedError
    else:
        output = _generate_no_beam_search(
            self,
            input_ids,
            inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            usr_token_id=usr_token_id,
            sys_token_id=sys_token_id,
            persona_token_id=persona_token_id,
            batch_size=effective_batch_size,
            model_kwargs=model_kwargs,
        )
    return output


@torch.no_grad()
def _generate_no_beam_search(
        self,
        input_ids,
        inputs_embeds,
        position_ids,
        attention_mask,
        token_type_ids,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        unk_token_id,
        pad_token_id,
        eos_token_id,
        cls_token_id,
        sep_token_id,
        usr_token_id,
        sys_token_id,
        persona_token_id,
        batch_size,
        model_kwargs,
):
    """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    output_ids = input_ids.new_zeros([input_ids.size(0), 0])
    original_input_ids = input_ids

    past_key_values = model_kwargs.pop('past_key_values', None)
    expand_vocab_size = model_kwargs.get('expand_vocab_size', None)

    gen_len = 0
    while gen_len < max_length:
        if inputs_embeds is None:
            prepared_input_ids = input_ids
            outputs = self.generate_step(
                input_ids=prepared_input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_cache=True,
                **model_kwargs,
            )
        else:
            outputs = self.generate_step(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_cache=True,
                **model_kwargs,
            )
        next_token_logits = outputs['lm_logits'][:, -1, :]
        if expand_vocab_size is not None:
            next_token_logits = next_token_logits[:, :-expand_vocab_size]

        scores = postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=torch.cat([original_input_ids, output_ids], dim=-1),
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=gen_len,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
            bos_token_id=bos_token_id,
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            usr_token_id=usr_token_id,
            sys_token_id=sys_token_id,
            persona_token_id=persona_token_id,
        )

        # if model has past, then set the past variable to speed up decoding
        past_key_values = outputs['past_key_values']

        if 'updated_kwargs' in outputs:
            model_kwargs.update(outputs.pop('updated_kwargs'))

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + eos_token_id * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = tokens_to_add.unsqueeze(-1)
        output_ids = torch.cat([output_ids, input_ids], dim=-1)

        position_ids = position_ids[:, -1:] + 1

        if model_kwargs.get('additive_embeds', None) is not None:
            model_kwargs['additive_embeds'] = model_kwargs.pop('additive_embeds')[:, -1:]

        if inputs_embeds is not None and token_type_ids is not None:
            token_type_ids = token_type_ids[:, 0].unsqueeze(1)
            inputs_embeds = None
        if attention_mask is not None:
            add_mask = attention_mask.new_ones([attention_mask.shape[0], 1], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, add_mask], dim=1)

        gen_len += 1
        unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

    return output_ids


def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        repetition_penalty,
        batch_size,
        num_beams,
        eos_token_id,
        bos_token_id=None,
        unk_token_id=None,
        pad_token_id=None,
        cls_token_id=None,
        sep_token_id=None,
        usr_token_id=None,
        sys_token_id=None,
        persona_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -1e20
    if pad_token_id is not None and cur_len < min_length:
        scores[:, pad_token_id] = -1e20
    if sep_token_id is not None and sep_token_id != eos_token_id:
        scores[:, sep_token_id] = -1e20
    if unk_token_id is not None and unk_token_id != eos_token_id:
        scores[:, unk_token_id] = -1e20
    if bos_token_id is not None and bos_token_id != eos_token_id:
        scores[:, bos_token_id] = -1e20
    if cls_token_id is not None and cls_token_id != eos_token_id:
        scores[:, cls_token_id] = -1e20
    if usr_token_id is not None:
        scores[:, usr_token_id] = -1e20
    if sys_token_id is not None:
        scores[:, sys_token_id] = -1e20
    if persona_token_id is not None:
        scores[:, persona_token_id] = -1e20

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len, scores.size(1)
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -1e5

    if bad_words_ids is not None:
        # Exclude EOS token (already processed)
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
        # Modify the scores in place by setting the banned tokens logits to `-inf`
        set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    return scores


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            if previous_token >= lprobs.size(1):
                continue
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int,
                             vocab_size: int):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            if any(e >= vocab_size for e in ngram):
                continue
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of banned tokens to ban in the format [[batch index, vocabulary position],...]
    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -1e5)
