import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import InputExample, RobertaForSequenceClassification, RobertaTokenizer, set_seed
from transformers import PreTrainedTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from utils.build import build_special_token_2_id, build_tokenizer_and_gpt2
from utils.common import build_path, filter_special_word, fix_sentence, norm, to_complete_sentence
from utils.constant import COMBINE_SPLITS, DOMAINS, DOMAIN_2_ID, RELS, SPLITS
from utils.cs import Comet, build_comet, do_expand


def extract_data_from_comae(file_dir: str, output_dir: str):
    print('Based on the speaker name to divide the data .')
    d = {
        f'{split}_{domain}': _extract_response_speaker_from_comae(f'{file_dir}/{split}_{domain}_annotated.txt')
        for split in SPLITS
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        same_in_all = d[f'train_{domain}'] & d[f'validation_{domain}'] & d[f'test_{domain}']
        same_in_train_validation = d[f'train_{domain}'] & d[f'validation_{domain}'] - same_in_all
        same_in_train_test = d[f'train_{domain}'] & d[f'test_{domain}'] - same_in_all
        same_in_validation_test = d[f'validation_{domain}'] & d[f'test_{domain}'] - same_in_all
        pure_train_speaker_name = d[f'train_{domain}'] - same_in_train_test - same_in_train_validation - same_in_all
        pure_validation_speaker_name = d[
                                           f'validation_{domain}'] - same_in_train_validation - same_in_validation_test - same_in_all
        pure_test_speaker_name = d[f'test_{domain}'] - same_in_train_test - same_in_validation_test - same_in_all

        _process_comae(file_dir, [f'train_{domain}_annotated.txt', f'validation_{domain}_annotated.txt',
                                  f'test_{domain}_annotated.txt'], same_in_all, output_dir,
                       f'{domain}_same_in_all.json')
        _process_comae(file_dir, [f'train_{domain}_annotated.txt', f'validation_{domain}_annotated.txt'],
                       same_in_train_validation, output_dir, f'{domain}_same_in_train_validation.json')

        _process_comae(file_dir, [f'train_{domain}_annotated.txt', f'test_{domain}_annotated.txt'], same_in_train_test,
                       output_dir, f'{domain}_same_in_train_test.json')

        _process_comae(file_dir, [f'validation_{domain}_annotated.txt', f'test_{domain}_annotated.txt'],
                       same_in_validation_test, output_dir, f'{domain}_same_in_validation_test.json')

        _process_comae(file_dir, [f'train_{domain}_annotated.txt'], pure_train_speaker_name, output_dir,
                       f'{domain}_pure_train.json')
        _process_comae(file_dir, [f'validation_{domain}_annotated.txt'], pure_validation_speaker_name, output_dir,
                       f'{domain}_pure_validation.json')
        _process_comae(file_dir, [f'test_{domain}_annotated.txt'], pure_test_speaker_name, output_dir,
                       f'{domain}_pure_test.json')


def _extract_response_speaker_from_comae(file_path: str) -> Set:
    speaker_name_set = set()
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), dynamic_ncols=True):
            dialog = json.loads(line)['dialog']
            for item in dialog:
                if item['speaker'] == 'sys':
                    speaker_name_set.add(item['speaker_name'])
    return speaker_name_set


def _process_comae(
        file_dir: str,
        file_name_list: List[str],
        speaker_name_set: Set,
        output_dir: str,
        output_file_name: str
):
    assert output_file_name.endswith('.json')
    f = open(f'{output_dir}/{output_file_name}', mode='w', encoding='utf-8')
    arr = []
    for file_name in file_name_list:
        with open(f'{file_dir}/{file_name}', mode='r', encoding='utf-8') as ff:
            for line in tqdm(ff.readlines(), dynamic_ncols=True):
                _line = json.loads(line)
                dialog = _line['dialog']
                for item in dialog:
                    if item['speaker'] == 'sys' and item['speaker_name'] in speaker_name_set:
                        arr.append(_line)
    json.dump(arr, f)
    f.close()
    print(f'{output_dir}/{output_file_name} saved')


def divide_overlap_speaker_data(file_dir: str, output_dir: str):
    file_name_list = [
        'pure_train', 'pure_validation', 'pure_test', 'same_in_all',
        'same_in_train_validation', 'same_in_train_test', 'same_in_validation_test'
    ]

    for domain in DOMAINS:
        d = {name: {'path': f'{file_dir}/{domain}_{name}.json'} for name in file_name_list}
        for name, value_dict in d.items():
            file_path = value_dict.get('path')
            with open(file_path, mode='r', encoding='utf-8') as f:
                data = json.load(f)
                if name.startswith('pure'):
                    value_dict["data"] = data  # List[Dict[str, List[str]]
                else:
                    assert name.startswith('same_in')
                    _data = defaultdict(list)
                    for item in tqdm(data, dynamic_ncols=True):  # type(item) Dict
                        for _item in item['dialog']:
                            if _item['speaker'] == 'sys':
                                _data[_item["speaker_name"]].append(item)
                                break
                    value_dict['data'] = _data

        # Divide "same_in_*.json" data as evenly as possible.
        # Ensure that the same "response_speaker" data can only appear in the same split
        for name, value_dict in d.items():
            if name.startswith('same_in'):
                groups_num = 3 if name == 'same_in_all' else 2
                split_result = _split_list(
                    [(speaker_name, len(data)) for speaker_name, data in value_dict.get('data').items()], groups_num)
                value_dict['split_result'] = split_result

        train = []
        validation = []
        test = []

        for name, value_dict in d.items():
            has_pure = 'pure' in name
            has_train = 'train' in name
            has_validation = 'validation' in name
            has_test = 'test' in name
            has_all = 'all' in name
            # if it has_pure, type of data is List[Dict[str, List[str]], else defaultdict
            data = value_dict.get('data')
            if has_pure:
                if has_train:
                    train.extend(data)
                elif has_validation:
                    validation.extend(data)
                else:
                    test.extend(data)
            else:
                if has_all:  # same_in_all
                    for speaker_name, feq in value_dict.get('split_result')[0]:
                        train.extend(data.get(speaker_name))
                    for speaker_name, feq in value_dict.get('split_result')[1]:
                        validation.extend(data.get(speaker_name))
                    for speaker_name, feq in value_dict.get('split_result')[2]:
                        test.extend(data.get(speaker_name))
                elif has_train and has_validation:
                    for speaker_name, feq in value_dict.get('split_result')[0]:
                        train.extend(data.get(speaker_name))
                    for speaker_name, feq in value_dict.get('split_result')[1]:
                        validation.extend(data.get(speaker_name))
                elif has_train and has_test:
                    for speaker_name, feq in value_dict.get('split_result')[0]:
                        train.extend(data.get(speaker_name))
                    for speaker_name, feq in value_dict.get('split_result')[1]:
                        test.extend(data.get(speaker_name))
                elif has_validation and has_test:
                    for speaker_name, feq in value_dict.get('split_result')[0]:
                        validation.extend(data.get(speaker_name))
                    for speaker_name, feq in value_dict.get('split_result')[1]:
                        test.extend(data.get(speaker_name))

        with open(f'{output_dir}/train_{domain}.json', mode='w', encoding='utf-8') as f:
            json.dump(train, f)
        print(f'{output_dir}/train_{domain}.json is saved')

        with open(f'{output_dir}/validation_{domain}.json', mode='w', encoding='utf-8') as f:
            json.dump(validation, f)
        print(f'{output_dir}/validation_{domain}.json is saved')

        with open(f'{output_dir}/test_{domain}.json', mode='w', encoding='utf-8') as f:
            json.dump(test, f)
        print(f'{output_dir}/test_{domain}.json is saved')


def _split_list(arr: List[Tuple[str, int]], groups_num: int, is_sorted=True):
    arr.sort(reverse=True, key=lambda t: t[1])
    result = defaultdict(list)
    arr_sum = sum(t[1] for t in arr)
    avg = arr_sum / groups_num
    for idx in range(groups_num):
        if idx == groups_num - 1:
            result[idx] = arr
            break
        if len(arr) and arr[0][1] >= avg:
            result[idx].append(arr[0])
            arr_sum -= arr[0][1]
            arr = arr[1:]
            avg = arr_sum / (groups_num - idx - 1)
        else:
            result[idx] = _get_list(arr, avg, avg)[0]
            for item in result[idx]:
                arr.remove(item)
    if not is_sorted:
        return list(result.values())
    sorted_idx_size = sorted([(idx, sum(t[1] for t in tuple_list)) for idx, tuple_list in result.items()],
                             key=lambda x: x[1], reverse=True)
    return [result.get(t[0]) for t in sorted_idx_size]


def _get_list(arr: List[Tuple[str, int]], need: float, distance: float):
    res = []
    if len(arr) == 0:
        return res, -1
    for i in range(len(arr) - 1):
        if need == arr[i]:
            res.append(arr[i])
            return res, 0
        elif need < arr[i][1]:
            continue
        else:  # need > arr[i]
            if i == 0:
                res.append(arr[i])
                need -= arr[i][1]
                tmp, d = _get_list(arr[i + 1:], need, need)
                res.extend(tmp)
                return res, d
            else:  # arr[i-1] > need > arr[i]
                dis1 = abs(arr[i - 1][1] - need)
                dis2 = abs(arr[i][1] - need)
                if dis1 > dis2:
                    res.append(arr[i])
                    need -= arr[i][1]
                    tmp, d = _get_list(arr[i + 1:], need, dis2)
                    res.extend(tmp)
                    return res, d
                else:
                    tmp, d = _get_list(arr[i:], need, dis2)
                    if dis1 > d:
                        res.extend(tmp)
                        return res, d
                    res.append(arr[i - 1])
                    return res, dis1
    dis = abs(need - arr[-1][1])
    if dis < distance:
        return arr[-1:], dis
    return [], -1


def extract_persona_from_pec(script, comae_dir, output_dir):
    print('Extracting persona info from pec corpus based on comae speaker......')
    comae_speaker = dict()
    for domain in DOMAINS:
        comae_domain_speaker = set()
        for split in SPLITS:
            comae_domain_speaker = comae_domain_speaker.union(
                _extract_response_speaker_from_comae(f'{comae_dir}/{split}_{domain}_annotated.txt'))
            comae_speaker[domain] = comae_domain_speaker

    comae_speaker_persona = dict()
    for domain in DOMAINS:
        d = dict()
        domain_dataset = load_dataset(script, name=domain)
        for split in SPLITS:
            print(f'Dealing with on the {domain}-{split}')
            for row in tqdm(domain_dataset[split], dynamic_ncols=True):
                if row['response_speaker'] in comae_speaker[domain]:
                    d[row['response_speaker']] = list(filter(
                        lambda s: s.strip() != 0,
                        [filter_special_word(s) for s in set(row['personas'])]
                    ))
                    comae_speaker_persona[DOMAIN_2_ID[domain]] = d
                    with open(f'{output_dir}/persona.json', 'w', encoding='utf-8') as f:
                        json.dump(comae_speaker_persona, f)
                    print(f'Saved as {output_dir}/persona.json')


def convert_data_to_input(file_dir: str, output_dir: str):
    for split in SPLITS:
        for domain in DOMAINS:
            exist_key = set()
            processed_data = []
            file_path = f'{file_dir}/{split}_{domain}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                for item in tqdm(json.load(f), dynamic_ncols=True,
                                 postfix=f'convert data to input: processing {file_path}'):
                    dialog = item['dialog']
                    assert dialog[-2]['speaker'] == 'usr'
                    assert dialog[-1]['speaker'] == 'sys'
                    ctx = fix_sentence(dialog[-2]['text'])
                    tgt = fix_sentence(dialog[-1]['text'])
                    if ctx == '' or tgt == '':
                        continue
                    if ctx + tgt in exist_key:
                        continue
                    exist_key.add(ctx + tgt)
                    save_dict = {
                        'domain': item['domain'],  # int
                        'speaker': dialog[-1]['speaker_name'],
                        'ctx': ctx,
                        'tgt': tgt,
                        'ctx_da': dialog[-2]['dialact'],
                        'ctx_em': dialog[-2]['emotion'],
                        'tgt_er': dialog[-1]['er'],
                        'tgt_ex': dialog[-1]['ex'],
                        'tgt_in': dialog[-1]['in'],
                        'tgt_da': dialog[-1]['dialact'],
                        'tgt_em': dialog[-1]['emotion']
                    }
                    processed_data.append(save_dict)

            with open(f'{output_dir}/{split}_{domain}_input.json', 'w', encoding='utf-8') as f:
                json.dump(processed_data, f)
                print(f'saved as {output_dir}/{split}_{domain}_input.json')


def _get_dataloader(tokenizer, input_examples, device):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        max_length=128,
        label_list=['0', '1'],
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader


def nli_persona(tokenizer, model, file_dir, output_dir, persona_data):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    for split in SPLITS:
        for domain in DOMAINS:
            processed_data = []
            file_path = f'{file_dir}/{split}_{domain}_input.json'
            for item in tqdm(json.load(open(file_path, 'r', encoding='utf-8')), dynamic_ncols=True):
                item['persona'] = []
                persona_info: List[str] = persona_data[str(item['domain'])][item['speaker']]
                cnt = 0
                samples = []
                for p in persona_info:
                    samples.append(InputExample(str(cnt), p, item['tgt'], '0'))
                    cnt += 1
                data_loader = _get_dataloader(tokenizer, samples, device)
                all_logits = None
                with torch.no_grad():
                    for batch in data_loader:
                        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                        outputs = model(**inputs)
                        if all_logits is None:
                            all_logits = outputs[0].detach()
                        else:
                            all_logits = torch.cat((all_logits, outputs[0].detach()), dim=0)
                results = torch.argmax(all_logits, dim=1)
                for i, r in enumerate(results):
                    if r == 2:
                        score = all_logits[i, r].item()
                        item['persona'].append((persona_info[i], score))
                if len(item['persona']) != 0:
                    # In descending order of score.
                    # If the scores are the same, they are sorted in descending order of sentence length
                    item['persona'].sort(key=lambda t: (t[1], len(t[0])), reverse=True)
                    item['persona'] = item['persona'][0][0]
                    processed_data.append(item)
            json.dump(processed_data, open(f'{output_dir}/{split}_{domain}_input_p.json', 'w', encoding='utf-8'))


def get_common_sense(model: Comet, file_dir: str, output_dir: str, **kwargs):
    for split in SPLITS:
        for domain in DOMAINS:
            file_path = f'{file_dir}/{split}_{domain}_input_p.json'
            processed_data = []
            with open(file_path, mode='r', encoding='utf-8') as f:
                for item in tqdm(json.load(f), dynamic_ncols=True, postfix=f'get cs: {file_path}'):
                    res_dict = do_expand(model, [item['ctx']], RELS, **kwargs)[0]
                    for key in res_dict:
                        if key != 'original':
                            item[f'ctx_{key}'] = res_dict[key]
                    processed_data.append(item)
            with open(f'{output_dir}/{split}_{domain}_input_p_cs.json', mode='w', encoding='utf-8') as f:
                json.dump(processed_data, f)
                print(f'expand persona by comet success and saved as {output_dir}/{split}_{domain}_input_p_cs.json')


def rewrite_persona_by_comet(
        model: Comet,
        file_dir: str,
        output_dir: str,
        **kwargs
):
    for split in SPLITS:
        for domain in DOMAINS:
            file_path = f'{file_dir}/{split}_{domain}_input_p_cs.json'
            processed_data = []
            with open(file_path, mode='r', encoding='utf-8') as f:
                for item in tqdm(json.load(f), dynamic_ncols=True, postfix=f'get cs: {file_path}'):
                    item['persona'] = do_expand(model, [item['persona']], RELS, **kwargs)[0]
                    processed_data.append(item)
            with open(f'{output_dir}/{split}_{domain}_input_p_cs_cs.json', mode='w', encoding='utf-8') as f:
                json.dump(processed_data, f)
                print(f'expand persona by comet success and saved as {output_dir}/{split}_{domain}_input_p_cs_cs.json')


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
            file_path = f'{file_dir}/{split}_{domain}_input_p_cs_cs.json'
            processed_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for item in tqdm(json.load(f), dynamic_ncols=True, postfix=f'convert {file_path}...'):
                    feature = {k: item[k] for k in
                               ['ctx_da', 'ctx_em', 'tgt_er', 'tgt_ex', 'tgt_in', 'tgt_da', 'tgt_em']}
                    ctx = fix_sentence(item['ctx'])
                    tgt = fix_sentence(item['tgt'])
                    if len(ctx) == 0 or len(tgt) == 0:
                        continue
                    feature.update({
                        'ctx_input_ids': tokenizer(ctx)['input_ids'][-max_ctx_len:] + [eos],
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


@hydra.main(version_base=None, config_path='config', config_name='prepare')
def main(cfg: DictConfig):
    cfg: dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    set_seed(cfg['seed'])
    output_dir = cfg['output']
    tmp_dir = build_path(output_dir, 'tmp_file')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    r"""
    Step 1: 
        Comae corpus were extracted and divided according to the responder. 
        Ensure that the same responder only appears in a split.
        Check the code to get output file path
    """
    extract_data_from_comae(cfg['comae_dir'], tmp_dir)
    divide_overlap_speaker_data(tmp_dir, tmp_dir)

    r"""
    Step 2:
        Extract the persona information from the pec base on comae corpus.
        It's independent of the previous step.
        You can execute this step before the extract_data_from_comae or after the divide_overlap_speaker_data.
        The output file is 'output/tmp_file/persona.json'
    """
    extract_persona_from_pec(script=cfg['pec_script'],
                             comae_dir=cfg['comae_dir'],
                             output_dir=tmp_dir)

    r"""
    Step 3:
        Change the corpus to json file. 
        The output file path is 'output/tmp_file/domain_split_input.json'
        domain in {happy, offmychest}; split in {train, validation, test}
    """
    convert_data_to_input(tmp_dir, tmp_dir)
    r"""
    Step 4:
        Use the NLI Model provided by https://github.com/caoyu-noob/D3 to filter persona sentence.
        Here, we only keep only the persona that is most relevant to response.
        After filtering,There is a case where there is no persona sentence associated with response. 
        For this part of the data, we throw it away.
        [data size]            after/original
        train_happy:            48083/78993
        train_offmychest:       38688/57618
        validation_happy:       11375/17463
        validation_offmychest:  7616/10706
        test_happy:             12441/19179
        test_offmychest:        7095/10302
    """
    persona_data = json.load(open(f'{tmp_dir}/persona.json', 'r', encoding='utf-8'))
    nli_tokenizer = RobertaTokenizer.from_pretrained(cfg['nli_path'])
    nli_model = RobertaForSequenceClassification.from_pretrained(cfg['nli_path'])
    nli_persona(tokenizer=nli_tokenizer,
                model=nli_model,
                file_dir=tmp_dir,
                output_dir=tmp_dir,
                persona_data=persona_data)

    r"""
    Step 5:
        Use COMET-BART to rewrite context to get common sense.
        Use COMET-BART to rewrite persona sentence to get more information
    """
    comet = build_comet(cfg['comet_path'])
    get_common_sense(
        model=comet,
        file_dir=tmp_dir,
        output_dir=tmp_dir,
        decode_method=cfg['decode_method'],
        num_generate=cfg['num_generate']
    )
    rewrite_persona_by_comet(
        model=comet,
        file_dir=tmp_dir,
        output_dir=tmp_dir,
        decode_method=cfg['decode_method'],
        num_generate=cfg['num_generate']
    )
    r"""
    Step 6:
        the last step is convert input to id.
        The output file path is split_domain.pkl.
        After combining split, we get train.pkl, validation.pkl, test_happy.pkl and test_offmychest.pkl
    """
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
