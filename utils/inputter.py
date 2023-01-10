import pickle

import torch
from torch.utils.data import Dataset

from utils.constant import RELS


class PECDataset(Dataset):
    def __init__(self, corpus_dir, split, domain=None):
        """
        Args:
            corpus_dir:
            split: only accept {'train', 'validation', 'test'}.
            domain(str): if the split is 'test', you should give domain as well.
        """
        super(PECDataset, self).__init__()

        if domain is not None:  # load the test dataset
            corpus_path = f'{corpus_dir}/{split}_{domain}.pkl'
        else:
            corpus_path = f'{corpus_dir}/{split}.pkl'

        with open(corpus_path, 'rb') as f:
            data = pickle.load(f)

        self.data = data
        self.data_size = len(data)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, tokenizer, do_gen=False):
    bs = len(batch)
    pad_token_id = tokenizer.pad_token_id

    forward_dict = {}
    forward_keys = ['ctx_input_ids', 'ctx_token_type_ids', 'ctx_da', 'ctx_em', 'ctx_len',
                    'tgt_input_ids', 'tgt_label_ids', 'tgt_token_type_ids', 'tgt_er',
                    'tgt_ex', 'tgt_in', 'tgt_da', 'tgt_em', 'tgt_len',
                    'persona_input_ids', 'persona_token_type_ids', 'persona_len'] + [f'ctx_{r}' for r in RELS]
    for key in forward_keys:
        if key in ['ctx_da', 'ctx_em', 'tgt_er', 'tgt_ex', 'tgt_in',
                   'tgt_da', 'tgt_em']:
            batch_inputs = torch.LongTensor([b[key] for b in batch]).view(bs, -1)
        elif key in ['ctx_len', 'tgt_len', 'persona_len']:
            batch_inputs = torch.LongTensor([b[key] for b in batch])
        else:
            # get the max len
            max_len = max(len(b[key]) for b in batch)
            # set padding value
            padding_value = pad_token_id if key != 'tgt_label_ids' else -1
            batch_inputs = torch.LongTensor([b[key] + [padding_value] * (max_len - len(b[key])) for b in batch])
        forward_dict[key] = batch_inputs

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


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x
