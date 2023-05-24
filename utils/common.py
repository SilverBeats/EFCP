import os
import re
from typing import Any, Dict, List

import emoji
import numpy as np
from torch import Tensor

from .constant import MARKS_MAP


def build_path(*args):
    size = len(args)
    p = str(args[0])
    for pp in range(1, size):
        p = os.path.join(p, str(args[pp]))
    return p


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


def data_2_device(data, device):
    if isinstance(data, dict):
        return {k: data_2_device(v, device) for k, v in data.items()}
    elif isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [data_2_device(item, device) for item in data]
    else:
        return data


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def config_to_dict(args) -> Dict[str, Any]:
    config_class = type(args)
    class_attrs = {k: v for k, v in vars(config_class).items() if not k.startswith('__')}
    instance_attrs = vars(args)
    d = {**class_attrs}
    d.update(**instance_attrs)
    return d


def check_model_config(model_name: str, ablation_mode: List[str], model_config: Dict[str, Any]):
    if model_name == 'multi':
        model_config['source_types'] = 2

    if model_name != 'efcp':
        return

    if ablation_mode is None:
        return

    if any(item not in ['pwp', 'per', 'cs', 'ef', 'pred_loss'] for item in ablation_mode):
        raise ValueError
    origin_model_config = {
        'use_persona': True,
        'use_cs': True,
        'use_ef': True,
        'predict_with_persona': True,
        'pred_comae_coef': 1.0
    }

    for item in ablation_mode:
        if item == 'pwp':
            origin_model_config['predict_with_persona'] = False
        elif item == 'per':
            origin_model_config['use_persona'] = False
            origin_model_config['predict_with_persona'] = False
        elif item == 'cs':
            origin_model_config['use_cs'] = False
        elif item == 'ef':
            origin_model_config['use_ef'] = False
            origin_model_config['predict_with_persona'] = False
            origin_model_config['pred_comae_coef'] = 0.0
        elif item == 'pred_loss':
            origin_model_config['pred_comae_coef'] = 0.0
        else:
            raise NotImplementedError

    model_config.update(origin_model_config)

    if not (0 < model_config['source_types'] <= 2):
        raise ValueError

    if model_config['attention_fusion_type'] not in ['sw', 'linear']:
        raise ValueError

    if model_config['use_persona']:
        model_config['source_types'] = 2
    else:
        model_config['source_types'] = 1
