import json
import logging
import os
import re
from typing import Any, Dict

import emoji
import numpy as np
import torch

from utils.constant import MARKS_MAP


def get_logger(name):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger(name)


def load_json_file(file_path: str):
    if not os.path.exists(file_path):
        raise ValueError(f'{file_path} does not exist')
    if not file_path.endswith('.json'):
        raise ValueError(f'{file_path} not a json file')
    with open(file_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def filter_special_word(s: str):
    """
    There are some special characters in the data set, such as \u263a, which we filter out
    There are some Chinese punctuation marks like \u201c \u201d which we change to English punctuation marks
    """
    s = re.sub(emoji.get_emoji_regexp(), "", s)  # remove the emoji
    for c, e in MARKS_MAP.items():
        s = s.replace(c, e)
    s = re.sub(r"[^a-zA-Z0-9,.:!?'\"\s()]", "", s)  # remove the special characters
    return ' '.join(s.strip().split())


def norm(text: str):
    return ' '.join(text.strip().split())


def fix_sentence(s):
    for k, v in MARKS_MAP.items():
        s = s.replace(k, v)
    s = norm(filter_special_word(s).lower())
    return s


def to_complete_sentence(sentence: str, effect_type):
    sentence = sentence.strip()
    if sentence == 'none' or sentence == '':
        return ''
    if effect_type == 'xAttr':
        sentence = 'i am ' + sentence
    elif effect_type == 'xEffect':
        if 'personax' not in sentence:
            sentence = 'i' + sentence
        else:
            sentence = sentence.replace('personax', 'i')
    elif effect_type in {'xIntent', 'xWant'}:
        sentence = 'i want ' + sentence
    elif effect_type == 'xNeed':
        sentence = 'i need ' + sentence
    elif effect_type == 'xReact':
        sentence = 'i feel ' + sentence
    else:
        # do not change sentence
        pass
    return sentence + ' .'


def calc_model_params(model):
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    return total_params


def save_model(model, save_path, additional_info: Dict[str, Any] = None, skip: List[str] = None):
    model_state_dict = {k: v.cpu() if v is not None else None  # save to cpu tensors
                        for k, v in model.state_dict().items()}
    if skip is not None:
        for item in skip:
            if item in model_state_dict:
                model_state_dict.pop(item)

    state_dict = {'model': model_state_dict}
    if additional_info is not None:
        state_dict.update(additional_info)
    torch.save(state_dict, save_path)
