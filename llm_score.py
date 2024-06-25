import json
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from utils.common import fix_sentence
from utils.eval import MyMetric

FILE_PATHS = {
    'gpt4-wo': 'output/限制长度17/gpt-4-wo_knowledge.csv',
    #'gpt4-with': 'output/限制长度+抽样/gpt-4-with_knowledge.csv'
}


def calc(df, start, end, metric_obj):
    metric_obj.refs = []
    metric_obj.hyps = []
    success_size = 0
    for i in range(start, end):
        try:
            metric_obj.forword(df.at[i, 'gold'], fix_sentence(df.at[i, 'target']))
            success_size += 1
        except Exception as e:
            continue

    return metric_obj.close(), success_size


@hydra.main(config_path="config", config_name='infer', version_base=None)
def main(config: DictConfig):
    config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    metric_config = config['infer']['metric_config']

    metric = MyMetric(**metric_config)
    # 加载数据
    split_size = 20000
    for _type, f_p in FILE_PATHS.items():
        output_dir = os.path.dirname(f_p)
        df = pd.read_csv(f_p)

        total_size = 0
        all_metric = {}

        success_size = 0
        for _idx, row in df.iterrows():
            try:
                metric.forword(row['gold'], fix_sentence(row['target']))
                success_size += 1
            except Exception as e:
                continue
        print(f'success_size = {success_size}')
        # n = 0
        # for i in range(0, df.shape[0], split_size):
        #     try:
        #         r, size = calc(df, i, min(df.shape[0], i + split_size), metric)

        #         total_size += size
        #         for k, v in r.items():
        #             if k not in all_metric:
        #                 all_metric[k] = 0
        #             all_metric[k] += v * size
        #         print(r)
        #     except Exception as e:
        #         error_batch = [{'gold': df.at[i, 'gold'], 'target': df.at[i, 'target']} for i in range(i, i + split_size)]
        #         with open(f'{output_dir}/{_type}_error_{n}.json', 'w', encoding='utf-8') as f:
        #             json.dump(error_batch, f, indent=4, ensure_ascii=False)
        #         n += 1

        # for k in all_metric.keys():
        #     all_metric[k] /= total_size
        r = metric.close()
        with open(f'{output_dir}/{_type}_metric.json', 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=4, ensure_ascii=False)
        print('=' * 40)


if __name__ == '__main__':
    main()
