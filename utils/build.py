from copy import deepcopy

from transformers import GPT2Tokenizer

from model import CEM, CoMAE, EFCP, MultiGPT2, Vanilla
from modeling import GPT2EncoderDecoderModel, MyGPT2LMHeadModel
from utils.constant import FUSE_C_P_TYPE, SPECIAL_TOKEN


def build_tokenizer_and_gpt2(name_or_path: str, only_tokenizer=False, **kwargs):
    tokenizer = GPT2Tokenizer.from_pretrained(name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKEN)
    if only_tokenizer:
        return tokenizer
    assert 'model_name' in kwargs and 'model_config' in kwargs
    model_name = kwargs['model_name']
    model_config = kwargs['model_config']
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


def build_model(gpt2, tokenizer, args):
    assert 'model_name' in args and 'model_config' in args

    model_name = args['model_name']
    model_config = args['model_config']

    if model_name != 'multi':
        token_id_dict = build_special_token_2_id(tokenizer)

    if model_name == 'efcp':
        model = EFCP(gpt2, token_id_dict, **model_config)
    elif model_name == 'comae':
        model = CoMAE(gpt2, token_id_dict, **model_config)
    elif model_name == 'vanilla':
        model = Vanilla(gpt2, token_id_dict, **model_config)
    elif model_name == 'multi':
        model = MultiGPT2(gpt2, **model_config)
    elif model_name == 'cem':
        model = CEM(gpt2, token_id_dict, **model_config)
    else:
        raise NotImplementedError()
    if model_name == 'efcp':
        use_persona = model_config['ablation_config']['use_persona']
        pred_with_persona = model_config['ablation_config']['pred_with_persona']
        fuse_c_p_type = model_config['fuse_c_p_type']
        assert fuse_c_p_type in FUSE_C_P_TYPE
        if not use_persona or not pred_with_persona:
            assert fuse_c_p_type == 'linear'
        model.set_fuse_module(k=model_config['k'])
    return model


def build_special_token_2_id(tokenizer):
    d = {
        'da': tokenizer.convert_tokens_to_ids('<da>'),
        'em': tokenizer.convert_tokens_to_ids('<em>'),
        'er': tokenizer.convert_tokens_to_ids('<er>'),
        'ex': tokenizer.convert_tokens_to_ids('<ex>'),
        'in': tokenizer.convert_tokens_to_ids('<in>'),
        'usr': tokenizer.convert_tokens_to_ids('<usr>'),
        'sys': tokenizer.convert_tokens_to_ids('<sys>'),
        'p': tokenizer.convert_tokens_to_ids('<persona>'),
        'pad': tokenizer.convert_tokens_to_ids('<pad>'),
        'eos': tokenizer.convert_tokens_to_ids('<eos>'),
        'cls': tokenizer.convert_tokens_to_ids('<cls>')
    }
    return d
