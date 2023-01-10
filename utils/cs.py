"""
coding=utf-8
@inproceedings{Hwang2021COMETATOMIC2O,
  title={COMET-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs},
  author={Jena D. Hwang and Chandra Bhagavatula and Ronan {Le Bras} and Jeff Da and Keisuke Sakaguchi and Antoine Bosselut and Yejin Choi},
  booktitle={AAAI},
  year={2021}
}
https://github.com/allenai/comet-atomic-2020/tree/master/models/comet_atomic2020_bart
here, we combine the generation_example.py and utils.py
"""

from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        print(f'using task specific params for {task}: {pars}')
        model.config.update(pars)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class Comet:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            num_generate=5,
            **kwargs
    ):
        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(
                    self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


ALL_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]


def get_comet(comet_ckpt_path: str):
    print(f'load the model from {comet_ckpt_path} ...')
    comet = Comet(comet_ckpt_path)
    return comet


def do_expand(comet: Comet, sentences: Iterable, rels: Iterable, **kwargs):
    assert set(rels).issubset(ALL_RELATIONS)
    comet.model.zero_grad()
    arr = []
    for sent in sentences:
        result_dict = {'original': sent}
        # head = f'{p} pleases ___ to make'
        queries = [f'{sent} {rel} [GEN]' for rel in rels]
        for rel, r in zip(rels, comet.generate(queries, **kwargs)):
            result_dict[rel] = r
        arr.append(result_dict)
    return arr
