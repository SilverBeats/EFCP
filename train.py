from functools import partial
from pprint import pp

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.build import build_tokenizer_and_plm, build_model
from utils.common import check_model_config, calc_model_params
from utils.eval import eval_model_loss
from utils.inputter import PECDataset, collate_fn
from utils.trainer import Trainer, TrainingArguments


@hydra.main(version_base=None, config_path='config', config_name='train')
def main(cfg: DictConfig):
    cfg: dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    model_name = cfg['model_name']
    train_config = cfg['train']
    model_config = cfg['model']
    ablation_mode = train_config.pop('ablation_mode')

    check_model_config(model_name, ablation_mode, model_config)
    print(f'Ablation_mode: {ablation_mode}')
    pp(cfg)

    if model_name in ['multi', 'efcp'] and model_config['share_embedding']:
        train_config['skip_param_save'] = [
            'plm.lm.weight',
            'plm.encoder.wpe.weight',
            'plm.encoder.wte.weight'
        ]

    corpus_dir = train_config.pop('corpus_dir')
    pretrained_model_path = train_config.pop('pretrained_model_path')

    print(f'load train data file from {corpus_dir} ...')
    train_data_set = PECDataset(corpus_dir, 'train')
    print(f'load validation data file from {corpus_dir} ...')
    valid_data_set = PECDataset(corpus_dir, 'validation')

    print(f'load pretrained model form {pretrained_model_path}...')
    tokenizer, plm = build_tokenizer_and_plm(pretrained_model_path, False, model_name, model_config)

    print(f'create {model_name} model')
    model = build_model(model_name, plm, tokenizer, model_config)
    print(f'Total parameter quantity: {calc_model_params(model, False)}')
    print(f'Trainable parameter quantity: {calc_model_params(model)}')

    train_args = TrainingArguments(**train_config)

    Trainer(
        model=model,
        config=train_args,
        train_dataset=train_data_set,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_cs=model_config['use_cs'],
            model_name=model_name
        ),
        eval_dataset=valid_data_set,
        eval_fn=eval_model_loss,
    ).train()


if __name__ == '__main__':
    main()
