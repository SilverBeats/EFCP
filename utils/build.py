from copy import deepcopy
from typing import Union

# from peft import LoraConfig, PeftType
from transformers import AutoTokenizer, GPT2Tokenizer, PretrainedConfig

from plm import (
    GPT2EncoderDecoderModel,
    MyGPT2LMHeadModel
)
from .constant import SPECIAL_TOKEN


def modify_tokenizer_model_config(obj: Union[PretrainedConfig, AutoTokenizer], tokenizer: AutoTokenizer):
    for key, value in SPECIAL_TOKEN.items():
        if key == 'additional_special_tokens':
            for item in value:
                setattr(obj, item[1:-1] + '_token', item)
                setattr(obj, item[1:-1] + '_token_id', tokenizer.convert_tokens_to_ids(item))
        else:
            setattr(obj, key, value)
            setattr(obj, key + '_id', tokenizer.convert_tokens_to_ids(value))


def build_tokenizer_and_gpt2(name_or_path: str, only_tokenizer=False, model_name=None, model_config=None):
    tokenizer = GPT2Tokenizer.from_pretrained(name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKEN)
    if only_tokenizer:
        return tokenizer

    model = GPT2EncoderDecoderModel.from_pretrained(name_or_path) \
        if model_name in ['efcp', 'multi'] \
        else MyGPT2LMHeadModel.from_pretrained(name_or_path)

    model.resize_token_embeddings(len(tokenizer))
    modify_gpt2_model(model, tokenizer, model_name, model_config)

    for key, value in SPECIAL_TOKEN.items():
        if key == 'additional_special_tokens':
            for item in value:
                setattr(model.config, item[1:-1] + '_token', item)
                setattr(model.config, item[1:-1] + '_token_id', tokenizer.convert_tokens_to_ids(item))
        else:
            setattr(model.config, key, value)
            setattr(model.config, key + '_id', tokenizer.convert_tokens_to_ids(value))
    return tokenizer, model


def modify_gpt2_model(model, tokenizer, model_name, model_config):
    model.config.summary_proj_to_labels = False
    if isinstance(model, MyGPT2LMHeadModel):
        return
    # set the encoder
    model.encoder = deepcopy(model.transformer)
    # encoder do not use context_attns and attention_module, so del them
    for block in model.encoder.h:
        if hasattr(block, 'context_attns'):
            del block.context_attns
        if hasattr(block, 'attention_module'):
            del block.attention_module

    # init the GPT2Model context_attn
    atten_fuse_type = model_config['attention_fusion_type']

    if model_name == 'efcp':
        use_persona = model_config['ablation_config']['use_persona']
        model_config['source_types'] = 2 if use_persona else 1

    if model_name == 'multi':
        model_config['source_types'] = 2

    for block in model.transformer.h:
        block.set_attention_pooling_module(atten_fuse_type, model_config['source_types'])
        block.set_context_attns(model_config['source_types'])
        for context_attn in block.context_attns:
            context_attn.load_state_dict(block.attn.state_dict())


def build_special_token_2_id(tokenizer):
    d = {
        'usr': tokenizer.convert_tokens_to_ids('<usr>'),
        'sys': tokenizer.convert_tokens_to_ids('<sys>'),
        'p': tokenizer.convert_tokens_to_ids('<persona>'),
        'pad': tokenizer.convert_tokens_to_ids('<pad>'),
        'eos': tokenizer.convert_tokens_to_ids('<eos>'),
    }
    return d


def build_model(model_name, plm, tokenizer, model_config):
    if model_name == 'efcp':
        from models import EFCP
        model = EFCP(plm, tokenizer, model_config)
    elif model_name == 'comae':
        from models import CoMAE
        model = CoMAE(plm, tokenizer)
    elif model_name == 'vanilla':
        from models import Vanilla
        model = Vanilla(plm, tokenizer)
    elif model_name == 'multi':
        from models import MultiGPT2
        model = MultiGPT2(plm, tokenizer)
    elif model_name == 'cem':
        from models import CEM
        model = CEM(plm, tokenizer, model_config)
    else:
        raise NotImplementedError()

    return model
