import json
import os
from functools import partial
from pprint import pp
from typing import Any, Dict
from typing import List

import hydra
import nltk
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from utils.build import build_model, build_tokenizer_and_gpt2
from utils.common import check_model_config, data_2_device
from utils.constant import ACC_LIST, DOMAINS
from utils.eval import MyMetric, eval_model_loss
from utils.inputter import PECDataset, collate_fn


def prepare_dataloader(
        args: Dict[str, Any],
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str,
        do_gen: bool,
        use_cs: bool
):
    dataset = PECDataset(args['corpus_dir'], split, None)
    batch_size = args['infer_batch_size']
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(
            __func=collate_fn,
            tokenizer=tokenizer,
            model_name=model_name,
            use_cs=use_cs,
            do_gen=do_gen,
        )
    )
    return dataloader


@hydra.main(version_base=None, config_path='config', config_name='infer')
def main(cfg: DictConfig):
    cfg: dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    model_name = cfg['model_name']
    generate_config = cfg['generate_config']
    infer_config = cfg['infer']
    model_config = cfg['model']
    ablation_mode = infer_config.pop('ablation_mode')

    check_model_config(model_name, ablation_mode, model_config)
    print(f'Ablation_mode: {ablation_mode}')
    pp(model_config)

    base_dir = os.path.dirname(infer_config['ckpt'])[:-5]
    infer_dir = os.path.join(base_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    device = infer_config['device']
    print(f'create {model_name} model and load ckpt ...')
    tokenizer, plm = build_tokenizer_and_gpt2(infer_config['gpt2_path'], model_name, model_config)

    model = build_model(model_name, plm, tokenizer, model_config).to(device)
    model.load_state_dict(torch.load(infer_config['ckpt'], map_location=device), strict=False)

    # save the config
    with open(f"{base_dir}/infer_config.json", 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False, sort_keys=False)

    output_dir = infer_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"load test data ...")
    valid_loader = prepare_dataloader(infer_config, model_name, tokenizer, 'test', False, model_config['use_cs'])
    infer_loader = prepare_dataloader(infer_config, model_name, tokenizer, 'test', True, model_config['use_cs'])

    metric_res = {}

    if model_name == 'comae' or (model_name == 'efcp' and model_config['use_ef']):
        # [right, total, acc]
        metric_res.update({item: [0, 0, 0.0] for item in ACC_LIST})
        # to save confusion_matrix
        confusion_matrix_dict = {item[:-5]: {'pred': [], 'right': []} for item in ACC_LIST}

    # to save the [post, reference, generation]
    res = []

    # do eval on test set
    eval_result = eval_model_loss(model, valid_loader, device)
    metric_res['perplexity'] = float(eval_result['ppl'])
    metric = MyMetric()

    for batch, posts, references in tqdm(infer_loader, desc='inferring'):
        batch = data_2_device(batch, device)
        bs = batch['ctx_input_ids'].shape[0]
        result_info, generations = model.generate(batch, **generate_config)

        if model_name == 'comae' or (model_name == 'efcp' and model_config['use_ef']):
            for item in ACC_LIST:
                metric_res[item][1] += bs
                right_list: List[int] = result_info[f'pred_{item[:-5]}'].view(-1).tolist()
                pred_list: List[List[int]] = result_info[f'pred_{item}'].tolist()
                right_cnt = sum(1 if r in p else 0 for r, p in zip(right_list, pred_list))
                metric_res[item][0] += right_cnt
                if item.endswith('top1'):
                    confusion_matrix_dict[item[:-5]]['pred'].extend(sum(pred_list, []))
                    confusion_matrix_dict[item[:-5]]['right'].extend(right_list)

        generations: List[str] = tokenizer.batch_decode(generations, skip_special_tokens=True)

        for p, r, g, persona in zip(posts, references, generations, batch['persona_input_ids']):
            metric.forword(r, g)
            res.append({
                'post': p,
                'response': r,
                'generation': g,
                'persona': tokenizer.decode(persona, skip_special_tokens=True)
            })

    with open(f'{output_dir}/gen.json', mode='w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4, sort_keys=False)

    print(f'Save the response generated by the model. You can find it at {output_dir}/gen.json')
    if model_name == 'comae' or (model_name == 'efcp' and model_config['use_ef']):
        for k, v in confusion_matrix_dict.items():
            with open(f'{output_dir}/{k}_pred_right.json', mode='w', encoding='utf-8') as f:
                json.dump(v, f, ensure_ascii=False, sort_keys=False)
            print(f'{output_dir}/{k}_pred_right.json saved')
            result = confusion_matrix(y_true=v['right'], y_pred=v['pred']).tolist()
            with open(f'{output_dir}/{k}_confusion_matrix.json', mode='w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, sort_keys=False)
            print(f'{output_dir}/{k}_confusion_matrix.json saved')

        for acc_item in ACC_LIST:
            right_cnt, total_cnt, _ = tuple(metric_res[acc_item])
            metric_res[acc_item][-1] = right_cnt / total_cnt

    print('Start of automatic evaluation')
    metric_res.update(metric.close())

    with open(f'{output_dir}/metric.json', mode='w', encoding='utf-8') as f:
        json.dump(metric_res, f, ensure_ascii=False, indent=4, sort_keys=True)
    print(f'Automatic evaluation result saved. You can find it at {output_dir}/metric.json')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
