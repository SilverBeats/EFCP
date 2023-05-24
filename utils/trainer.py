import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from .common import build_path, config_to_dict, data_2_device, make_infinite


def create_dataloader(dataset, batch_size, collate_fn=None, shuffle=False, drop_last=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


@dataclass
class TrainingArguments:
    experiment: str = field(
        default='default',
        metadata={'help': '当前实验名称, 创建输出目录时会使用到'}
    )
    output_dir: str = field(
        default='output',
        metadata={'help': '输出目录'}
    )
    seed: int = field(
        default=7,
        metadata={'help': '随机种子'}
    )
    epochs: int = field(
        default=5,
        metadata={'help': '迭代的总轮数'}
    )
    optim_steps: Optional[int] = field(
        default=None,
        metadata={'help': '总优化步数, 当与epochs同时设定时, optim_steps起作用'}
    )
    device: str = field(
        default='cpu',
        metadata={'help': '运行设备'}
    )
    train_batch_size: int = field(
        default=32,
        metadata={'help': '训练批大小'}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={'help': '梯度裁剪'}
    )
    warm_up: Union[int, float] = field(
        default=0.1,
        metadata={'help': '值小于1, warm_up_step为总优化步数的10%; 值大于等于1, warm_up_step为warm_up'}
    )
    lr: float = field(
        default=1e-4,
        metadata={'help': '学习率'}
    )
    cache_empty_step: int = field(
        default=20,
        metadata={'help': '清缓存的间隔'}
    )
    loss_label: str = field(
        default='loss',
        metadata={'help': '模型返回结果是一个字典, 损失的key名'}
    )
    do_eval: bool = field(
        default=False,
        metadata={'help': '是否在训练期间进行验证'}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={'help': '验证批大小'}
    )
    eval_interval: int = field(
        default=2000,
        metadata={'help': '验证的间隔'}
    )
    eval_key_label: str = field(
        default='loss',
        metadata={'help': 'eval_fn返回一个字典, 验证结果具有决定性的指标名'}
    )
    low_is_better: bool = field(
        default=True,
        metadata={'help': 'eval_key_label是越小越好还是越大越好'}
    )
    ckpt: str = field(
        default=None,
        metadata={'help': '加载保存点, 断点接续训练'}
    )
    skip_param_save: List[str] = field(
        default=None,
        metadata={'help': '保存模型时需要跳过的参数'}
    )
    keep_one: bool = field(
        default=True,
        metadata={'help': '训练完成之后是否只保存最好的那个保存点'}
    )
    do_log: bool = field(
        default=True,
        metadata={'help': '是否日志记录'}
    )
    train_log_items: List[str] = field(
        default=None,
        metadata={'help': '训练时需要记录的字段名'}
    )
    eval_log_items: List[str] = field(
        default=None,
        metadata={'help': '验证时需要记录的字段名'}
    )


class Trainer:
    OPTIMIZER = 'optimizer'
    SCHEDULE = 'schedule'
    TRAIN_STATE = 'train_state'
    MODEL = 'model'

    def __init__(
            self,
            model: nn.Module,
            config: TrainingArguments,
            train_dataset: Dataset,
            collate_fn: Callable = None,
            eval_dataset: Dataset = None,
            eval_fn: Callable = None
    ):
        # check for training
        assert config.epochs is not None or config.optim_steps is not None
        assert config.ckpt is not None or config.output_dir is not None
        assert train_dataset is not None
        if config.epochs and config.optim_steps:
            print('When given both epochs and optim_step, '
                  'the program uses the latter to calculate '
                  'the total number of optimization steps .')
        # check for evaluating
        if config.do_eval:
            assert eval_dataset is not None, 'You should give eval_dataset because do_eval is equal to True.'
            assert eval_fn is not None, 'You should give a custom eval_fn because do_eval is equal to True.'
            assert config.eval_interval > 0
            assert config.eval_key_label is not None, 'You should give eval_key_label because do_eval is equal to True.'
        if config.skip_param_save is None:
            config.skip_param_save = []

        # check for logging
        if config.do_log:
            config.train_log_items = config.train_log_items if config.train_log_items is not None else []
            config.eval_log_items = config.eval_log_items if config.eval_log_items is not None else []

        assert config.seed is not None
        set_seed(config.seed)

        self.model = model
        self.config = config

        # create dataloader
        self.train_loader = create_dataloader(train_dataset, config.train_batch_size, collate_fn, True)
        if config.do_eval:
            self.eval_fn = eval_fn
            self.eval_loader = create_dataloader(eval_dataset, config.eval_batch_size, collate_fn, False)
        # create optimizer, lr schedule
        config.optim_steps = config.optim_steps if config.optim_steps else len(self.train_loader) * config.epochs
        warm_up_steps = config.warm_up if config.warm_up >= 1 else config.warm_up * config.optim_steps
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        self.schedule = get_linear_schedule_with_warmup(self.optimizer, warm_up_steps, config.optim_steps)

        # prepare some variable
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.base_dir = build_path(config.output_dir, config.experiment, current_time)

        INF = 1e9
        if not config.low_is_better:
            INF = - INF

        self.train_state = {
            'step': 0,
            'best_key_label_value': INF,
            'best_ckpt_path': ''
        }

        # load ckpt if necessary
        if config.ckpt:
            self._load_ckpt(config.ckpt)

        self.ckpt_dir = build_path(self.base_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if config.do_log:
            self.log_dir = build_path(self.base_dir, 'log')
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=2)

        # save the config
        with open(build_path(self.base_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_to_dict(config), f, indent=2)

    def _load_ckpt(self, ckpt):
        assert os.path.exists(self.config.ckpt)
        # update the base_dir
        self.base_dir = os.path.dirname(ckpt)[:-5]
        # load weights
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict[self.MODEL], strict=False)
        self.optimizer.load_state_dict(state_dict[self.OPTIMIZER])
        self.schedule.load_state_dict(state_dict[self.SCHEDULE])
        self.train_state.update(state_dict[self.TRAIN_STATE])
        self.train_state['best_ckpt_path'] = ckpt

    def _log(self, result: dict, item_list: List[str], tag: str):
        self.writer.add_scalars(main_tag=tag, tag_scalar_dict={
            k: v.item() if isinstance(v, Tensor) else v for k, v in result.items()
            if k in item_list
        }, global_step=self.train_state['step'])

    def _save_model(self, eval_result: dict):
        key_label_value = eval_result[self.config.eval_key_label]
        if isinstance(key_label_value, Tensor):
            key_label_value = key_label_value.item()
        if (
                (self.config.low_is_better and key_label_value < self.train_state['best_key_label_value'])
                or
                (not self.config.low_is_better and key_label_value > self.train_state['best_key_label_value'])
        ):
            if self.config.keep_one and os.path.exists(self.train_state['best_ckpt_path']):
                os.remove(self.train_state['best_ckpt_path'])

            step = self.train_state['step']
            best_ckpt_path = build_path(self.ckpt_dir, f'step_{step}.pt')

            self.train_state.update({
                'best_key_label_value': key_label_value,
                'best_ckpt_path': best_ckpt_path
            })

            model_state_dict = self.model.state_dict()

            for item in self.config.skip_param_save:
                if item in model_state_dict:
                    model_state_dict.pop(item)

            torch.save({
                self.TRAIN_STATE: self.train_state,
                self.OPTIMIZER: self.optimizer.state_dict(),
                self.SCHEDULE: self.schedule.state_dict(),
                self.MODEL: model_state_dict
            }, best_ckpt_path)

            print(f'Model saved. You can find it at {best_ckpt_path}')

    def _clean_ckpt(self):
        r"""
        After training, delete the optimizer, schedule and train_state, only keep the model
        """
        state_dict = torch.load(self.train_state['best_ckpt_path'])
        model_weight = state_dict[self.MODEL]
        try:
            os.remove(self.train_state['best_ckpt_path'])
            torch.save(model_weight, self.train_state['best_ckpt_path'])
        except Exception as e:
            print(f'something goes wrong: {e}. _clean_ckpt() failed !')
            if not os.path.exists(self.train_state['best_ckpt_path']):
                torch.save(state_dict, self.train_state['best_ckpt_path'])

    def train(self):
        self.model.to(self.config.device)
        train_pbar = tqdm(total=self.config.optim_steps, dynamic_ncols=True, desc='training', leave=True)

        train_loader = make_infinite(self.train_loader)
        if self.train_state['step'] != 0:
            train_pbar.update(self.train_state['step'])
            for _ in range(self.train_state['step']):
                next(train_loader)
        while True:
            self.model.train()
            batch = data_2_device(next(train_loader), self.config.device)
            outputs: dict = self.model(**batch)

            loss = outputs[self.config.loss_label]
            loss.backward()
            self.optimizer.step()
            self.schedule.step()
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # do log
            if self.config.do_log:
                self._log(outputs, self.config.train_log_items, 'train')

            # update
            self.train_state['step'] += 1
            train_pbar.update(1)

            show_dict = {}
            for k, v in outputs.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    show_dict[k] = round(v.item(), 2)
                if isinstance(v, int) or isinstance(v, float):
                    show_dict[k] = round(v, 2)
            train_pbar.set_postfix(show_dict)

            if self.config.do_eval and self.train_state['step'] % self.config.eval_interval == 0:
                self.model.eval()
                eval_result = self.eval_fn(self.model, self.eval_loader, self.config.device)
                self.model.train()
                if self.config.do_log:
                    self._log(eval_result, self.config.eval_log_items, 'eval')
                self._save_model(eval_result)

            if self.train_state['step'] % self.config.cache_empty_step == 0:
                torch.cuda.empty_cache()

            if self.train_state['step'] == self.config.optim_steps:
                break

        if self.config.do_eval and self.train_state['step'] % self.config.eval_interval != 0:
            self.model.eval()
            eval_result = self.eval_fn(self.model, self.eval_loader, self.config.device)
            self.model.train()
            if self.config.do_log:
                self._log(eval_result, self.config.eval_log_items, 'eval')
            self._save_model(eval_result)

        if self.config.do_log:
            self.writer.close()
        train_pbar.close()
        self._clean_ckpt()
