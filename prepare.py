import json
import os
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import set_seed

from utils.build import build_tokenizer_and_gpt2
from utils.common import build_path, fix_sentence, norm, to_complete_sentence
from utils.constant import COMBINE_SPLITS, DOMAINS, RELS, SPLITS


def convert_input_to_ids(
        tokenizer: PreTrainedTokenizer,
        file_dir: str,
        output_dir: str,
        **kwargs
):
    max_ctx_len = kwargs.get('max_ctx_len', 150)
    max_tgt_len = kwargs.get('max_tgt_len', 40)
    eos = tokenizer.eos_token_id
    usr = tokenizer.usr_token_id
    sys = tokenizer.sys_token_id
    p = tokenizer.persona_token_id

    for split in SPLITS:
        for domain in DOMAINS:
            file_path = f'{file_dir}/{split}_{domain}.json'
            processed_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f'convert {file_path}...')
                for item in tqdm(json.load(f), dynamic_ncols=True):
                    feature = {
                        k: item[k] for k in
                        ['ctx_da', 'ctx_em', 'tgt_er', 'tgt_ex', 'tgt_ip', 'tgt_da', 'tgt_em']
                    }
                    ctx = fix_sentence(item['ctx'])
                    tgt = fix_sentence(item['tgt'])
                    persona = fix_sentence(item['persona']['original'])
                    if len(ctx) == 0 or len(tgt) == 0 or len(persona) == 0:
                        continue
                    feature.update({
                        'ctx_input_ids': tokenizer(ctx)['input_ids'][-max_ctx_len:],
                        'tgt_label_ids': tokenizer(tgt)['input_ids'][-max_tgt_len:] + [eos]
                    })
                    feature['ctx_len'] = len(feature['ctx_input_ids'])
                    feature['ctx_token_type_ids'] = [usr] * feature['ctx_len']

                    feature['tgt_input_ids'] = [eos] + feature['tgt_label_ids'][:-1]
                    feature['tgt_len'] = len(feature['tgt_input_ids'])
                    feature['tgt_token_type_ids'] = [sys] * feature['tgt_len']

                    persona_cs_sentences = []
                    for rel in RELS:
                        persona_sentence = ' '.join(to_complete_sentence(s, rel) for s in item['persona'][rel])
                        persona_cs_sentences.append(persona_sentence)
                    persona = norm(' '.join(persona_cs_sentences))
                    feature['persona_input_ids'] = tokenizer(persona)['input_ids']
                    feature['persona_len'] = len(feature['persona_input_ids'])
                    feature['persona_token_type_ids'] = [p] * feature['persona_len']

                    for rel in RELS:
                        feature[f'ctx_{rel}'] = list(
                            filter(lambda s: s.strip() != 'none', map(str.lower, item[f'ctx_{rel}'])))
                        feature[f'ctx_{rel}'] = sum(tokenizer(feature[f'ctx_{rel}'])['input_ids'], [])

                    processed_data.append(feature)
            with open(f'{output_dir}/{split}_{domain}.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
                print(f'saved as {output_dir}/{split}_{domain}.pkl')


def combine_split(file_dir: str, output_dir: str):
    for split in SPLITS:
        data = []
        for domain in DOMAINS:
            with open(f'{file_dir}/{split}_{domain}.pkl', mode='rb') as f:
                data.extend(pickle.load(f))
            if split not in COMBINE_SPLITS:
                with open(f'{output_dir}/{split}_{domain}.pkl', mode='wb') as f:
                    pickle.dump(data, f)
                    print(f'{output_dir}/{split}_{domain}.pkl is saved')
                    data = []
        if split in COMBINE_SPLITS:
            with open(f'{output_dir}/{split}.pkl', mode='wb') as f:
                pickle.dump(data, f)
                print(f'{split} combine success, and {output_dir}/{split}.pkl is saved')


def process_dataset(file_dir: str, output_dir: str):
    r"""处理原始数据集, ctx的常识句子只保留一句"""
    for split in SPLITS:
        for domain in DOMAINS:
            file_name = f'{split}_{domain}.json'
            file_path = build_path(file_dir, file_name)
            save_path = build_path(output_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f'processing {file_path}')
                process_data = []
                for item in tqdm(json.load(f), dynamic_ncols=True):
                    for rel in RELS:
                        key = f'ctx_{rel}'
                        if key in item:
                            cs = [v for v in item[key] if v.lower().strip() != 'none']
                            item[key] = [cs[0]]
                    process_data.append(item)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(process_data, f)


@hydra.main(version_base=None, config_path='config', config_name='prepare')
def main(cfg: DictConfig):
    cfg: dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    set_seed(cfg['seed'])
    output_dir = cfg['output_dir']
    tmp_dir = build_path(output_dir, 'tmp_file')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    process_dataset(cfg['corpus_dir'], tmp_dir)

    gpt2_tokenizer = build_tokenizer_and_gpt2(cfg['gpt2_path'], True)
    convert_input_to_ids(
        tokenizer=gpt2_tokenizer,
        file_dir=tmp_dir,
        output_dir=tmp_dir,
        max_ctx_len=cfg['max_ctx_len'],
        max_tgt_len=cfg['max_tgt_len']
    )
    combine_split(tmp_dir, output_dir)


if __name__ == '__main__':
    main()
