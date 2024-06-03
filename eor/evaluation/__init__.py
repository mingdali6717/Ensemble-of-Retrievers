from .metrics import gpt3_judge, llm_evalutor, f1_score, rouge1_score, rouge2_score, rougeL_score, bleu_score, em_score, BemCalculator, RareWordF1Calculator

from .utils import ResultSaver, load_datasets
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

EVALUATOR_MAPPING = {
    "judger": gpt3_judge,
    "llm": llm_evalutor,
    "f1": f1_score,
    "em": em_score,
    "rouge1": rouge1_score,
    "rouge2": rouge2_score,
    "rougeL": rougeL_score,
    "bleu": bleu_score,
}

METRIC_NAME_MAPPING = {
    "llm": "llm_acc",
    "f1": "f1_score",
    "em": "em_score",
    "rouge1": "rouge1_score",
    "rouge2": "rouge2_score",
    "rougeL": "rougeL_score",
    "bleu": "bleu_score",
    "bem": "bem_score",
    "raref1": "raref1_score"
}

METRIC_MAPPING = {
    "NQ": ["em", "bem", "f1"],
    "WQ": ["em", "bem", "f1"],
    "TQ": ["em", "bem", "f1"],
    "TruthfulQ": ["llm", "judger", "bleu", "rouge1", "bem", "rougeL", "rouge2", "raref1"]
}

CACHED_BEM_PATH = 'data/model/bem/answer_equivalence_bem_1'

def metric_max_over_ground_truths(question, candidate, ground_truth, metric_func):
    scores_for_ground_truths = []
    if candidate == '':
        candidate = "None"
    for g in ground_truth:
        if g == '':
            score = 0.0
        else:
            score = metric_func(candidate, g)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class  Evaluator:

    def __init__(self, dataset_name, metrics=None):
        assert dataset_name in METRIC_MAPPING, f"supported datasets are: {', '.join(list(METRIC_MAPPING.keys()))}. "
        self.dataset_name = dataset_name
        
        if metrics is None:
            self.metrics = METRIC_MAPPING[dataset_name]
        else: 

            for metric in metrics:
                assert metric in METRIC_MAPPING[dataset_name], f" metric '{metric}' is not supported for dataset {dataset_name}"
            self.metrics = metrics
        
        


    
    def evaluate(self, path_or_data, max_num=None, cached_score_path: str = None, cached_score_name_mapping = None, cached_score = None, system_only=False, verbose=True, batch_size=512):
        """
        parameters:
        path_or_data: str or Dict -  the path or data itself.  data should be in the format of:
        {
        query: {
                "system_answer" : str,
                "system_answer_name": str,
                "ground_truth_answer": List[str]
                "details":{
                    "method_name_1": {
                        "knowledge": str,
                        "response": str,
                        "vote_score": float,
                        "final_score": float,
                        "reward_score": float,(optional) 
                    }
                }
        
            }
        }
        cached_score_path: str - file path where save the cached metric score
        cached_score: dict - cached_score constructed by self.load_cached_score, if given, cached_score_path will be ignored.
        cached_score_name_mapping: dict - key is one of the supported metric name, value is the column name in file corresponding to the key
        system_only: bool - if true, will only evaluate the system performance, else, will evaluate all score of each method.
        """
        if cached_score is not None:
            if cached_score_name_mapping is None:

                raise KeyError("Please proved a dictionary which key is the default metric names, and the value is the corresponing metric score column name in the cached score data")
            else:
                assert isinstance(cached_score, dict), "cached_score should be a dict constructed by self.load_cached_score"
                
            metrics_to_load = set(self.metrics).intersection(set(cached_score_name_mapping.keys()))
            if metrics_to_load != set():
                if verbose:
                    print(f"metrics to be loaded from cached file are: {'|'.join(list(metrics_to_load))} ")
                metrics_to_load_mapping = {name: cached_score_name_mapping[name] for name in list(metrics_to_load)}
                self.cached_scores = cached_score
            else:
                metrics_to_load = set()


        elif cached_score_path is not None:
            assert os.path.exists(cached_score_path), f"given cached score path '{cached_score_path}' do not exists"
            if cached_score_name_mapping is None:
                raise KeyError("Please proved a dictionary which key is the default metric names, and the value is the corresponing metric score column name in the cached file")
            metrics_to_load = set(self.metrics).intersection(set(cached_score_name_mapping.keys()))
            if metrics_to_load != set():
                if verbose:
                    print(f"metrics to be loaded from cached file are: {'|'.join(list(metrics_to_load))} ")
                metrics_to_load_mapping = {name: cached_score_name_mapping[name] for name in list(metrics_to_load)}
                self.load_cached_score(cached_score_path, metrics_to_load_mapping, verbose=verbose)
        else:
            
            metrics_to_load = set()
        
        if metrics_to_load == set():
            print("no cached metric score found, start to evaluate from scratch")

        if "raref1" in self.metrics and "raref1" not in metrics_to_load:
            raref1_corpus_path = ["eor/evaluation/metrics/test.jsonl", "eor/evaluation/metrics/validation.jsonl", "eor/evaluation/metrics/train.jsonl"]
            raref1scorer = RareWordF1Calculator(corpus_path=raref1_corpus_path, top_p=0.1)
            EVALUATOR_MAPPING["raref1"] = raref1scorer.raref1_score
        
        if "bem" in self.metrics and "bem" not in metrics_to_load:
            model_path = CACHED_BEM_PATH
            bem_scorer = BemCalculator(model_path=model_path)
            EVALUATOR_MAPPING["bem"] = bem_scorer.bem_score
        
            
        self.questions, self.knowledges, self.candidates, self.ground_truth, self.methods = load_datasets(path_or_data)
        if system_only:
            self.candidates = {"system_answer": self.candidates["system_answer"]}
            self.knowledges = {"system_answer": self.knowledges["system_answer"]}
            self.methods = {"system_answer": self.methods["system_answer"]}
        if max_num is not None:
            self.questions = self.questions[:max_num]
            for k,v in self.knowledges.items():
                self.knowledges[k] = v[:max_num]
            for k,v in self.candidates.items():
                self.candidates[k] = v[:max_num]
            for k,v in self.methods.items():
                self.methods[k] = v[:max_num]
            self.ground_truth = self.ground_truth[:max_num]
            
        if verbose:
            print(f"start to evaluate dataset {self.dataset_name} with metrics: {', '.join(self.metrics)}.")
        ground_truth = ["; ".join(gt) for gt in self.ground_truth]

        results = dict()

        for method, ans in self.candidates.items():
            
            method_result = dict()
            for metric in self.metrics:
                if verbose:
                    print(f"evaluate answer generated by {method} with metric {metric}")
                if metric in metrics_to_load:
                    methods = self.methods[method]
                    method_result.update(self.evaluate_one_metric_from_cache(self.questions, ans, methods, metric))
                else:
                    method_result.update(self.evaluate_one_metric(self.questions, ans, self.ground_truth, metric, batch_size=batch_size))
            knowledge = self.knowledges.get(method, None)  # 如果没有匹配的知识，默认为 None
            method_result.update({"questions": self.questions, "knowledges": knowledge, "answers": ans, "ground_truth": ground_truth, "methods": self.methods[method]})
            results[method] = method_result
        self.result = ResultSaver(results, verbose=verbose)
        return self.result, results
        #return self.result

    def evaluate_one_metric_from_cache(self, questions, candidates, methods, metric):
        scores = []
        
        for q, c, m in zip(questions, candidates, methods):
            key = self.key(q, c, m)
            
            scores.append(self.cached_scores[key][metric])
            
        return {METRIC_NAME_MAPPING[metric]: scores}
    
    def load_cached_score(self, path, metric_mapping, verbose=True):
        """
        path: str - file path where save the cached metric score
        metric_mapping: dict - key is one of the supported metric name, value is the column name in file corresponding to the key
        """
        if verbose:
            print(f"load cached score from {path}")
        data = pd.read_excel(path)
        data.fillna("", inplace=True)
        data_dict = data.to_dict(orient="records")
        self.cached_scores = dict()
        column_mapping = {v:k for k,v in metric_mapping.items()}



        for ex in data_dict:

            key = self.key(ex["questions"],ex["answers"], ex["models"])
            self.cached_scores[key] = dict()

            for c_name in column_mapping.keys():

                self.cached_scores[key][column_mapping[c_name]] = ex[c_name]
        
        return self.cached_scores


    def key(self, question, answer, method):
        #return question + "[sep]" + answer + "[sep]" + method
        return question + "[sep]" + method



    def evaluate_one_metric(self, questions, candidates, ground_truths, metric, batch_size=512):
        """
        input:
        questions: List[str]
        candidates: List[str]
        ground_truths: List[List[str]]
        metric: str

        return:
        dict{"{score_name}": scores (List[float])}
        """
        assert metric in EVALUATOR_MAPPING, f"metric should be one of:{';'.join(list(EVALUATOR_MAPPING.keys()))}"
        if metric == "judger":
            score, acc = EVALUATOR_MAPPING["judger"](questions, candidates)
            return {"judger_score": score, "judger_acc": acc}
        elif metric in ["f1", "rouge1", "bleu", "em", "rouge2", "rougeL", "raref1"]:
            scores = []
            metric_func = EVALUATOR_MAPPING[metric]
            for q, c, g in tqdm(zip(questions, candidates, ground_truths)):
                
                score = metric_max_over_ground_truths(q, c, g, metric_func)
                scores.append(score)
            return {METRIC_NAME_MAPPING[metric]: scores}
        elif metric in ["bleu"]:
            scores = []
            metric_func = EVALUATOR_MAPPING[metric]
            for c, g in tqdm(zip(candidates, ground_truths)):
                
                score = metric_func(c, g)
                scores.append(score)
            return {METRIC_NAME_MAPPING[metric]: scores}
        elif metric == "bem":
            scores = []
            expand_qs = []
            expand_cs = []
            expand_gt = []
            index = [0]
            count_index = 0
            for q,c, gs in zip(questions, candidates, ground_truths):
                if c == "":
                    c = "None"
                for g in gs:
                    count_index += 1
                    expand_qs.append(q)
                    expand_cs.append(c)
                    expand_gt.append(g)
                index.append(count_index)
            examples = [{
                    'question': q,
                    'reference': g,
                    'candidate': c
                } for q, g, c in zip(expand_qs, expand_gt, expand_cs)]
            
            if "bem" not in EVALUATOR_MAPPING:
                model_path = CACHED_BEM_PATH
                bem_scorer = BemCalculator(model_path=model_path)
                EVALUATOR_MAPPING["bem"] = bem_scorer.bem_score
        
            expand_scores = EVALUATOR_MAPPING[metric](examples, batch_size=batch_size)
            scores = [max(expand_scores[index[i]: index[i+1]]) for i in range(len(index)-1)]
            return {METRIC_NAME_MAPPING[metric]: scores}
    
        else:
            score = EVALUATOR_MAPPING[metric](questions, ground_truths, candidates)
            return {METRIC_NAME_MAPPING[metric]: score}
    
    def doc_analysis(self, path_or_data, max_num=None, verbose=True):
        self.questions, self.knowledges, self.candidates, self.ground_truth, self.methods = load_datasets(path_or_data)
        if max_num is not None:
            self.questions = self.questions[:max_num]
            for k,v in self.knowledges.items():
                self.knowledges[k] = v[:max_num]
            for k,v in self.candidates.items():
                self.candidates[k] = v[:max_num]
            for k,v in self.methods.items():
                self.methods[k] = v[:max_num]
            self.ground_truth = self.ground_truth[:max_num]
        ground_truth = ["; ".join(gt) for gt in self.ground_truth]
        results = dict()
        for method, ans in self.candidates.items():
            
            method_result = dict()
            
            knowledge = self.knowledges.get(method, None)

            if knowledge is None:
                continue
            
    
            method_result.update({"doc_gt_em": self.doc_em(knowledge, self.ground_truth) })
            method_result.updata({"doc_ans_em": self.doc_em(knowledge, ans)})
              # 如果没有匹配的知识，默认为 None
            method_result.update({"questions": self.questions, "knowledges": knowledge, "answers": ans, "ground_truth": ground_truth, "methods": self.methods[method]})
            results[method] = method_result
        return ResultSaver(results, verbose=verbose)
    
    def doc_em(self, knowledges, answers):
        if isinstance(answers[0], list) and isinstance(answers[0][0], str):
            scores = [metric_max_over_ground_truths(None, k, a, em_score) for k, a in zip(knowledges, answers)]
        elif isinstance(answers[0], str):
            scores = [em_score(k, a) for k, a in zip(knowledges, answers)]
        else:
            raise KeyError("something wrong with answrs")
        
        return scores




        
    
    
        




