import pickle
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .constant import RELS


class PECDataset(Dataset):
    def __init__(self, corpus_dir: str, split: str, domain: str = None):
        super(PECDataset, self).__init__()
        assert split in {'train', 'validation', 'test'}
        if split == 'test':
            assert domain is not None
            corpus_path = f'{corpus_dir}/{split}_{domain}.pkl'
        else:
            corpus_path = f'{corpus_dir}/{split}.pkl'

        with open(corpus_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data_size = len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        return self.data[index]

    def __len__(self) -> int:
        return self.data_size


def collate_fn(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        do_gen: bool = False,
        use_cs: bool = True
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], List[str], List[str]]]:
    bs = len(batch)
    pad_token_id = tokenizer.pad_token_id
    usr_token_id = tokenizer.usr_token_id

    forward_dict = {}
    forward_keys = ['ctx_da', 'ctx_em',
                    'ctx_len', 'tgt_input_ids', 'tgt_label_ids', 'tgt_token_type_ids',
                    'tgt_er', 'tgt_ex', 'tgt_ip', 'tgt_da', 'tgt_em', 'tgt_len',
                    'persona_input_ids', 'persona_token_type_ids', 'persona_len']
    if not use_cs:
        forward_keys.extend(['ctx_input_ids', 'ctx_len', 'ctx_token_type_ids'])
    for key in forward_keys:
        if key in ['ctx_da', 'ctx_em', 'tgt_er', 'tgt_ex', 'tgt_ip', 'tgt_da', 'tgt_em']:
            batch_inputs = torch.LongTensor([b[key] for b in batch]).view(bs, -1)  # (bs, 1)
        elif key.endswith('len'):
            batch_inputs = torch.LongTensor([b[key] for b in batch])  # (bs, )
        else:
            # get the max len
            max_len = max(len(b[key]) for b in batch)
            # set padding value
            padding_value = pad_token_id if key != 'tgt_label_ids' else -1
            batch_inputs = torch.LongTensor([b[key] + [padding_value] * (max_len - len(b[key])) for b in batch])
        forward_dict[key] = batch_inputs
    if use_cs:
        ctx_input_ids = []
        ctx_len = []
        for b in batch:
            cc = []
            for rel in RELS:
                if rel == 'xAttr':
                    continue
                if f'ctx_{rel}' in b:
                    cc.append(b[f'ctx_{rel}'])
            cc.append(b['ctx_input_ids'])
            ctx_input_ids.append(cc)
            ctx_len.append(len(cc))
        # padding
        max_ctx_len = max(ctx_len)
        # (bs, max_ctx_len)
        ctx_input_ids = torch.LongTensor([
            item + [pad_token_id] * (max_ctx_len - l)
            for item, l in zip(ctx_input_ids, ctx_len)
        ])
        # (bs, max_ctx_len)
        ctx_token_type_ids = torch.LongTensor([
            [usr_token_id] * l + [pad_token_id] * (max_ctx_len - l) for l in ctx_len
        ])
        ctx_len = torch.LongTensor(ctx_len)  # (bs, )
        forward_dict['ctx_input_ids'] = ctx_input_ids
        forward_dict['ctx_token_type_ids'] = ctx_token_type_ids
        forward_dict['ctx_len'] = ctx_len

    if do_gen:
        handler_keys = ['tgt_input_ids', 'tgt_label_ids', 'tgt_token_type_ids']
        for key in handler_keys:
            forward_dict[key] = forward_dict[key][:, 0].unsqueeze(1)
        forward_dict['tgt_len'] = torch.ones((bs,), dtype=torch.long)

        posts = tokenizer.batch_decode([b['ctx_input_ids'] for b in batch], skip_special_tokens=True)
        references = tokenizer.batch_decode([b['tgt_label_ids'] for b in batch], skip_special_tokens=True)
        return forward_dict, posts, references
    else:
        return forward_dict
