import json

import nltk
from bert_score import score
from nlgeval import NLGEval
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

do_nlgeval = True
do_bert_score = False
do_sim_score = False

sim_pretrained_path = r'J:\pretrained_models\all_mpnet-base-v2'
number_layers = 17
model_type = './pre_trained/roberta-large/'
batch_size = 2
root_dir = f"output/eos!=pad/ef/infer"

if do_nlgeval:
    nlgeval = NLGEval(no_skipthoughts=True)

if do_sim_score:
    sim_model = SentenceTransformer(sim_pretrained_path).cuda()

# for domain in DOMAINS:
# target_dir = f'{root_dir}/{domain}'
target_dir = root_dir
hyps = []
refs = []
with open(f'{target_dir}/metric.json', mode='r', encoding='utf-8') as f:
    metric_res = json.load(f)
with open(f'{target_dir}/gen.json', mode='r', encoding='utf-8') as f:
    data = json.load(f)
    for item in tqdm(data, desc=f'loading {target_dir}/gen.json'):
        ref = ' '.join(nltk.word_tokenize(item['response'].lower()))
        hyp = ' '.join(nltk.word_tokenize(item['generation'].lower()))
        refs.append(ref)
        hyps.append(hyp)
if do_nlgeval:
    metrics_dict = nlgeval.compute_metrics(ref_list=[refs], hyp_list=hyps)
    metric_res.update(metrics_dict)
if do_bert_score:
    P, R, F = score(
        cands=hyps,
        refs=refs,
        lang='en',
        batch_size=batch_size,
        num_layers=number_layers,
        model_type=model_type
    )
    metric_res.update({
        'bert_score_P': round(P.mean().item(), 6),
        'bert_score_R': round(R.mean().item(), 6),
        'bert_score_F': round(F.mean().item(), 6),
    })
if do_sim_score:
    embeddings1 = sim_model.encode(hyps, convert_to_tensor=True, device='cuda')
    embeddings2 = sim_model.encode(refs, convert_to_tensor=True, device='cuda')
    sim_score = util.pairwise_cos_sim(embeddings1, embeddings2).mean().item()
    metric_res.update({'sim_score': sim_score})

# overwrite
with open(f'{target_dir}/metric.json', mode='w', encoding='utf-8') as f:
    json.dump(metric_res, f, ensure_ascii=False, indent=4, sort_keys=False)
