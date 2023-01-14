import json
import os
from datetime import datetime
from functools import partial

import torch
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from utils.build import build_model, build_tokenizer_and_gpt2
from utils.common import calc_model_params, get_logger, load_json_file, save_model
from utils.constant import CACHE_EMPTY_STEP, INF, SKIP_SAVE_PARAMS
from utils.eval import eval_model_loss
from utils.inputter import PECDataset, collate, make_infinite

logger = get_logger(__name__)
best_loss = INF
best_ppl = INF
best_ckpt_path = ''
step = 0

args = load_json_file('config/train.json')
set_seed(args['seed'])
args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
args['device'] = 'cpu'
tokenizer, gpt2 = build_tokenizer_and_gpt2(name_or_path=args['gpt2_path'],
                                           model_name=args['model_name'],
                                           model_config=args['model_config'])
# prepare the dataset and dataloader
train_set = PECDataset(args['corpus_dir'], 'train')
train_sampler = RandomSampler(train_set)
train_loader = DataLoader(
    dataset=train_set,
    sampler=train_sampler,
    batch_size=args['train_batch_size'],
    collate_fn=partial(collate, tokenizer=tokenizer),
    drop_last=True
)
valid_set = PECDataset(args['corpus_dir'], 'validation')
valid_sampler = SequentialSampler(valid_set)
valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=args['valid_batch_size'],
    sampler=valid_sampler,
    collate_fn=partial(collate, tokenizer=tokenizer),
    drop_last=True
)

# prepare the model, optimizer and scheduler
logger.info(f"build the {args['model_name']} model...")

model = build_model(gpt2, tokenizer, args)
model.to(args['device'])
logger.info(f'model parameters={calc_model_params(model)}')

optimizer = AdamW(model.parameters(), lr=args['lr'])

if args['epochs'] > 0:
    args['optim_steps'] = len(train_loader) * args['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=args['warmup_steps'],
                                            num_training_steps=args['optim_steps'])

if args['ckpt']:  # load the ckpt
    args['base_dir'] = os.path.dirname(args['ckpt'])[:-5]
    args['ckpt_dir'] = os.path.join(args['base_dir'], 'ckpt')
    args['log_dir'] = os.path.join(args['base_dir'], 'log')
    os.makedirs(args['log_dir'], exist_ok=True)
    os.makedirs(args['ckpt_dir'], exist_ok=True)

    logger.info(f"Load ckpt: {args['ckpt']}")
    state_dict = torch.load(args['ckpt'])
    model.load_state_dict(state_dict.pop('model'), strict=False)
    optimizer.load_state_dict(state_dict.pop('optimizer'))
    scheduler.load_state_dict(state_dict.pop('scheduler'))
    step = state_dict['step']
    if step != 0:
        for _ in range(step):
            next(train_loader)
else:  # training from scratch
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    args['base_dir'] = os.path.join(args['output_dir'], f"{args['model_name']}_{current_time}")
    args['ckpt_dir'] = os.path.join(args['base_dir'], 'ckpt')
    args['log_dir'] = os.path.join(args['base_dir'], 'log')
    os.makedirs(args['log_dir'], exist_ok=True)
    os.makedirs(args['ckpt_dir'], exist_ok=True)
    with open(f"{args['base_dir']}/train_config.json", 'w', encoding='utf-8') as f:
        json.dump(args, f, indent=4, ensure_ascii=False, sort_keys=False)

eval_logger = open(f"{args['log_dir']}/eval.csv", 'a+', buffering=1)
print('step,eval_loss,eval_ppl', file=eval_logger)
pbar = tqdm(total=args['optim_steps'], desc=f'training', dynamic_ncols=True)
pbar.update(step)


def eval_and_save_model(file_name):
    global best_ckpt_path
    global best_loss
    global best_ppl

    model.eval()
    eval_loss, eval_ppl = eval_model_loss(model, valid_loader, args['device'])
    model.train()

    if eval_loss < best_loss:
        additional_info = {
            'step': step,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_path = os.path.join(args['ckpt_dir'], f'{file_name}.pt')
        save_model(model.module if args['distributed'] else model, save_path, additional_info, SKIP_SAVE_PARAMS)
        logger.info(f"{save_path} save succeed")

        # remove the worse one
        if os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
        best_ckpt_path = save_path
        best_loss = eval_loss
        best_ppl = eval_ppl
    logger.info(f'best_loss={best_loss}')
    logger.info(f'best_ppl={best_ppl}')
    logger.info(f'best_ckpt_path={best_ckpt_path}')
    return eval_loss, eval_ppl


train_loader = make_infinite(train_loader)
while True:
    model.train()
    batch = {k: v.to(args['device']) if isinstance(v, Tensor) else v for k, v in next(train_loader).items()}
    bs = batch['ctx_input_ids'].shape[0]
    outputs = model(**batch)
    loss = outputs.pop('all')
    loss = loss / bs
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    step += 1
    pbar.set_postfix(ordered_dict={k: round(v.item(), 2) for k, v in outputs.items()})
    pbar.update(1)

    # do eval and save model, optimizer and scheduler
    if step % args['eval_step_interval'] == 0:
        (
            eval_loss,
            eval_ppl
        ) = eval_and_save_model(model, valid_loader, args, step, optimizer, scheduler, str(step))
        print(f'{step},{eval_loss},{eval_ppl}', file=eval_logger)
        logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))

    if step % CACHE_EMPTY_STEP == 0:
        torch.cuda.empty_cache()

    if step >= args['optim_steps']:
        break

if step >= args['optim_steps'] and args['epochs'] > 0 and step % args['eval_step_interval'] != 0:
    (
        eval_loss,
        eval_ppl
    ) = eval_and_save_model(model, valid_loader, args, step, optimizer, scheduler, str(step))
    print(f'{step},{eval_loss},{eval_ppl}', file=eval_logger)
    logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))
