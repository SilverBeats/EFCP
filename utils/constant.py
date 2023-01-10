DOMAINS = ['happy', 'offmychest']
DOMAIN_2_ID = {'happy': 0, 'offmychest': 1}
SPLITS = ['train', 'validation', 'test']
COMBINE_SPLITS = ['train', 'validation']

MARKS_MAP = {
    '\u3002': '.',
    '\uff1b': ';',
    '\uff0c': ',',
    '\uff1a': ':',
    '\u201c': '"',
    '\u201d': '"',
    '\uff08': '(',
    '\uff09': ')',
    '\u3001': ',',
    '\uff1f': '?',
    '\u2018': "'",
    '\u2019': "'"
}

RELS = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

FUSE_C_P_TYPE = ['linear', 'co']

SPECIAL_TOKEN = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'cls_token': '<cls>',
    'unk_token': '<unk>',
    'sep_token': '<sep>',
    'additional_special_tokens': ['<usr>', '<sys>', '<persona>', '<er>', '<ex>', '<in>', '<da>', '<em>']
}

INF = 100000000
CACHE_EMPTY_STEP = 25
SKIP_SAVE_PARAMS = ['gpt2.lm_head.weight']

ACC_LIST = ['er_top1', 'ex_top1', 'in_top1',
            'da_top1', 'da_top3',
            'em_top1', 'em_top3']