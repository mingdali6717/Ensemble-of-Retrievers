from ..utils import LLM
import math
import re
import string
import torch
from bert_score import bert_cos_score_idf, get_idf_dict, sent_encode
from functools import partial
from itertools import chain
from multiprocessing import Pool, get_context
from collections import Counter, defaultdict
from transformers import DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from math import log

def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    if nthreads > 0:
        with get_context("spawn").Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
            p.close()
            p.join()
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict


def BertScore(model_name, hyps, refs, device, batch_size=64):
    """
    Calculate bert score with idf weighted
    args:
    model_name: str -  base model for embedding caculation, should be one of supported model name in LLMConfig
    hyps: List[str] - candidate sentences
    refs: List[str] - reference sentences
    device: str - device to use, should be in the format of f"gpu{device_id}" or 'cpu'

    return: torch.tensor - dim(hyp_len, 3), each vector in dim=1 are P, R, F score for each hyp, ref pair
    """
    llm_name = LLM.initial_lm(model_name, device)
    model, tokenizer = LLM.llms[llm_name]
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

   
    
    num_layers = 40 if model_name == "deberta" else 8
        
    if len(model.encoder.layer) > num_layers:
        model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
    elif len(model.encoder.layer) < num_layers:
        assert False, "Model layer num error"

    
    idf_dict = get_idf_dict(hyps, tokenizer)
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0  
    

    bert_scores = bert_cos_score_idf(model, refs, hyps, tokenizer, idf_dict, device=device.replace("gpu","cuda:"), batch_size=batch_size, verbose=True).cpu()

    
    return bert_scores



def RareBertScore(model_name, hyps, refs, device, batch_size=64, Corpus=None, top_p=0.5):
    if Corpus is None: 
        Corpus = hyps
    llm_name = LLM.initial_lm(model_name, device)
    model, tokenizer = LLM.llms[llm_name]
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    num_layers = 40 if model_name == "deberta" else 8
        
    if len(model.encoder.layer) > num_layers:
        model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
    elif len(model.encoder.layer) < num_layers:
        assert False, "Model layer num error"
    
    process_partial = partial(process, tokenizer=tokenizer)
    idf_count = Counter()
    token_count = Counter()
    
    with get_context("spawn").Pool(4) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, hyps)))
        p.close()
        p.join()
    if Corpus != hyps:
        with get_context("spawn").Pool(4) as p:
            token_count.update(chain.from_iterable(p.map(process_partial, Corpus)))
    else:
        token_count = idf_count
    
    
    idf_dict = defaultdict(lambda: math.log((len(hyps) + 1) / (1)))
    idf_dict.update(
        {idx: math.log((len(hyps) + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )

    target = sum(token_count.values()) * top_p
    cumul = 0
    for token, freq in token_count.most_common():
        cumul += freq
        if cumul > target:
            break
        else:
            idf_dict[token] = 0.0

    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0  

    bert_scores = bert_cos_score_idf(model, refs, hyps, tokenizer, idf_dict, device=device.replace("gpu","cuda:"), batch_size=batch_size, verbose=True).cpu()

    
    return bert_scores

def NLIScore(model_name, hyps, refs, device, batch_size=64, queries = None, with_query=False):
    if with_query:
        assert queries is not None and len(queries) == len(hyps), f"queries should be given and with the same length as hyps and refs"
        hyps = [q+" "+h for q,h in zip(queries, hyps)]
        refs = [q+" "+r for q,r in zip(queries, refs)]
    llm_name = LLM.initial_lm(model_name, device)
    model, tokenizer = LLM.llms[llm_name]
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    scores = calculate_nli_score(model, tokenizer, hyps, refs, batch_size=batch_size)
    entailment_score = torch.tensor([s["entailment"] for s in scores], dtype=torch.double)
    return entailment_score


def EMScore(hyps, refs):
    scores = [em_score(hyp, ref) for hyp, ref in zip(hyps, refs)]
    return torch.tensor(scores, dtype=torch.double)
def F1Score(hyps, refs):
    scores = [f1_score(ref, hyp) for hyp, ref in zip(hyps, refs)]
    return torch.tensor(scores, dtype=torch.double)
    

def pooling_score(score_tensor, pooling_method='max', threshold=None, min_acceptance_num=None, topk=None):
    """
    flatten the score matrix along with the dim=1.  s_i_j is the similarity scroe of sentence i,j, we want pooling {s_i_j}_j to a scalar s_i
    Parameters:
    score_tensor: torch.tensor(n * n-1), s_i_j is the similarity score of sentence i and sentence j
    pooling_method: str - one of max, mean, topk, max, voting, majority_voting.
        - max: take max_j {s_i_j} as the pooling score s_i
        - mean: take mean_j{s_i_j} as the pooling score s_i
        - topk: take mean(topk_j{s_i_j}) as the pooling score s_i
        - voting: take I_i * mean(topk_j{s_i_j}) as the pooling score s_j, where I_i = 1 if count(s_i_j > threshold) >= k, 0 otherwise, k is the min_acceptance_num
        - majority_voting: I_i * mean(topk_j{s_i_j}), where I_i = 1 if count_i({s_i_j > threshold}_j) = max_i(count_i({s_i_j > threshold}_j)), k is the max_i(count_i({s_i_j > threshold}_j))
    
    threshold: float - threshold to filter out answer, default is 0.5
    min_acceptance_num: int - when pooling_method is voting, only answer with more than 'min_accepatance_num' similar answers which similarity score > threshold will be kept. if None, min_acceptance_num will be set to math.ceil(total_num/2)-1
    topk: int - if pooling method is topk, will only take the average of the top 'mean_pooling_topk' number of scores, default is math.ceil(num/2)

    """
    score_size = score_tensor.shape[1]
    majority_num = max(1, math.ceil((score_size+1)/2))-1

    
    if score_tensor.dim() == 1:

        return score_tensor
    else:
        if pooling_method == "mean":
            return score_tensor.mean(dim=1, dtype=torch.double)
        elif pooling_method == "topk":
            
            if topk==None:
                topk = majority_num
            else:
                topk = min(topk, score_size)
            
            return score_tensor.topk(topk, dim=1)[0].mean(dim=1, dtype=torch.double)
        elif pooling_method == "max":
            return score_tensor.max(dim=1)[0]
        elif pooling_method == "voting":
            assert threshold is not None, "threshold should be given for voting method"
    
            if min_acceptance_num is None:
                min_acceptance_num = majority_num
            else:
                min_acceptance_num = min(min_acceptance_num, majority_num)
            
            identifier = ((score_tensor >= threshold).sum(dim=1) >= min_acceptance_num).double()
            score = score_tensor.topk(min_acceptance_num, dim=1)[0].mean(dim=1, dtype=torch.double)
            return (score * identifier)
        elif pooling_method == "majority_voting":
            assert threshold is not None, "threshold should be given for voting method"
            similar_num = (score_tensor >= threshold).sum(dim=1)
            max_similar = torch.max(similar_num).item()
            identifier = (similar_num >= max_similar)
            score = score_tensor.topk(max(max_similar, 1), dim=1)[0].mean(dim=1, dtype=torch.double)
            return (score * identifier)

        else:
            raise KeyError(f"flatten method '{pooling_method}' is not supported, please use 'max', 'mean' or 'voting")

def calculate_nli_score(model, tokenizer, hyps, refs, batch_size=64):
    """
    calculate the element-wise entailment score of hyps and refs
    Parameters:
    model: nn.Module - nli model used for calculatiton
    tokenizer:
    hyps; List[str]
    refs: List[str]
    batch_size: int
    
    return:
    scores: List[dict()] - the key of each dict is "entailment", "neutral" and "contradiction",  value is the correponding normalized predicted probability.
    """
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
    
    concated_sents = [hyp + "[$usedforsep$]" + ref for hyp, ref in zip(hyps, refs)]

    sents = dedup_and_sort(concated_sents)
    compact_hyps, compact_refs = list(zip(*[s.split("[$usedforsep$]") for s in sents]))


    
    if "deberta-v2-xlarge-mnli" in  model.name_or_path or "deberta-xlarge-mnli" in model.name_or_path:
        label_names = ["contradiction", "neutral", "entailment"]
    else:
        label_names = ["entailment", "neutral", "contradiction"]
    tokenized_input = tokenizer(compact_hyps, compact_refs, truncation=True)
    with torch.no_grad():
        data_collator = DataCollatorWithPadding(tokenizer,  pad_to_multiple_of=8 if model.dtype==torch.float16 else None)
        dataloader = DataLoader(Dataset.from_dict(tokenized_input.data), batch_size=batch_size, collate_fn=data_collator)
        compact_scores = []
        for batch in tqdm(dataloader):
            output = model(**batch.to(model.device))
            score = torch.softmax(output["logits"], dim=-1).tolist()
            if len(score[0]) == 2:
                label_names = [label_names[0], label_names[-1]]
            prediction = [{name: pred for pred, name in zip(s, label_names)} for s in score]
            compact_scores.extend(prediction)
    
    stat_dict = {k:v for k,v in zip(sents, compact_scores)}
    scores = [stat_dict[k] for k in concated_sents]
    
        
    return scores


def gather_bidirectional_nli_score(scoring_matrix, gathering_method='mean'):
    """
    scoring_list: torch.tensor(n, n-1) - in the size n*n-1, a similarity matrix except values in the diagonal. this function is used to gatter score_i_j and score_j_i in the original matrix
    """

    
    res_num = scoring_matrix.size(0)
    if res_num == 1:
        return scoring_matrix
    score_list = scoring_matrix.reshape(res_num*(res_num-1)).tolist()
    gathered_score = []
    for i in range(res_num):
        for j in range(res_num-1):
            if j>=i:
                i_transpose = j+1
            else:
                i_transpose = j
            j_transpose = i
            if j_transpose > i_transpose:
                j_transpose -= 1
        
            if gathering_method == 'mean':
                score = (score_list[i*(res_num-1) + j] + score_list[i_transpose*(res_num-1) + j_transpose])/2
            elif gathering_method == 'max':
                score = max(score_list[i*(res_num-1) + j], score_list[i_transpose*(res_num-1) + j_transpose])
            else:
                AttributeError(f"given gathering method '{gathering_method}' is not supported, should be one of mean or max ")
            gathered_score.append(score)
    
    return torch.tensor(gathered_score, dtype=torch.double).view(res_num, res_num-1)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def em_score(candidate, reference):
    candidate = normalize_answer(candidate)
    reference = normalize_answer(reference)
    pattern = r'\b' + re.escape(reference) + r'\b'
    if re.search(pattern, candidate):
        return 1
    else:
        return 0


    

