from collections import defaultdict
from functools import partial, reduce
from itertools import chain
from ..base import BaseVoter
from ..utils import BertScore, RareBertScore, NLIScore, EMScore, F1Score, pooling_score,gather_bidirectional_nli_score
from loguru import logger
import json


import torch


METHODS_LIST = ["nli", "bertscore", "rarebertscore", "em", "f1", "nli_with_query"]

EN_NLI_MODEL_NAME = "deberta_nli"
EN_BERT_SCORE_MODEL_NAME = "deberta"


class StandardVoter(BaseVoter):
    def __init__(self, config, result_dict=None):
        super().__init__(config, result_dict=result_dict)
        self.use_method_weight = config["method_weight"]
        self.threshold =  config["threshold"] if config["threshold"] is not None else 0.5
        self.pooling_threshold = config["pooling_threshold"] if config["pooling_threshold"] is not None else 0.8
        assert self.threshold >=0 and self.threshold <=1, f"threshold should be number between [0, 1], but '{self.threshold} is given'"

        
        scoring_method = config["scoring_method"] if config["scoring_method"] is not None else 'nli'
        assert scoring_method in METHODS_LIST or scoring_method == "composition", f"given scroing method '{scoring_method}' is not supported, please give one of {'|'.join(METHODS_LIST)} or 'composition'"
        composition_weight = config["composition_weight"] if config["composition_weight"] is not None else [1.0,0.0,0.0,0.0,0.0]
        composition_weight = [float(w)/sum(composition_weight) for w in composition_weight]
        
        self.bidirectional = [n for n, bo in config["bidirectional"].items() if bo]
        self.hierarchical_voting = config["hierarchical_voting"]["turn_on"]
        if scoring_method == "composition":
            assert len(composition_weight) == len(METHODS_LIST), f"compostion weight should be the same length as supported methods, but '{composition_weight}' are given, which sum to {sum(composition_weight)}"
            to_print = json.dumps({m: w for m,w in zip(METHODS_LIST, composition_weight)}, indent=4)
            if self.verbose:
                logger.info(f"Using 'compositon' method for voting, voting weight are:\n{to_print}")
            self.scoring_method = [m for m,w in zip(METHODS_LIST, composition_weight) if w > 0]
            self.composition_weight = {m: w for m,w in zip(METHODS_LIST, composition_weight) if w > 0}
        else:
            self.scoring_method = [scoring_method]
            self.composition_weight = {scoring_method: 1.0}

        self.pooling_method = config["pooling_method"] if config["pooling_method"] is not None else "max"
        if self.pooling_method in ["voting", "majority_voting"]:
            assert self.pooling_threshold >=0 and self.pooling_threshold <=1, f"threshold should be number between [0, 1], but '{self.pooling_threshold} is given'"
        self.min_acceptance_num = config["min_acceptance_num"]
        self.topk = config["mean_pooling_topk"]
        self.batch_size = config["batch_size"]
        if self.hierarchical_voting:
            self.h_voting = dict()
            self.h_voting["pooling_method"] = config["hierarchical_voting"]["pooling_method"] if config["hierarchical_voting"]["pooling_method"] is not None else self.pooling_method
            self.h_voting["pooling_threshold"] = config["hierarchical_voting"]["pooling_threshold"] if config["hierarchical_voting"]["pooling_threshold"] is not None else self.pooling_threshold
            self.h_voting["min_acceptance_num"] = config["hierarchical_voting"]["min_acceptance_num"] if config["hierarchical_voting"]["min_acceptance_num"] is not None else self.min_acceptance_num
            self.h_voting["topk"] = config["hierarchical_voting"]["mean_pooling_topk"] if config["hierarchical_voting"]["mean_pooling_topk"] is not None else self.topk
        else: 
            self.h_voting = False
        
        
        self.config = config
        self.config["threshold"] = self.threshold
        
        self.config["pooling_threshold"] = self.pooling_threshold
        self.config["pooling_method"] = self.pooling_method
        
        
    
    
    def _batch_voting(self, queries, language_list, responses_list, device="gpu0", voting_method=None):

        if voting_method is None:
            scoring_method = self.scoring_method
        else:
            assert voting_method in METHODS_LIST, f"given '{voting_method}' is not supported"
            scoring_method = [voting_method]
        if self.verbose:
            logger.info(f"{device}: start to vote with voting methods: {scoring_method}")

        if "zh" in language_list and "en" not in language_list:
            model_name = "bert-base-chinese"
            scoring_method = ["bertscore"]
            self.composition_weight = {"bertscore": 1.0}
            if self.verbose:
                logger.warning("chinese only support bertscore")
        elif "zh" not in language_list and "en" in language_list:

            if scoring_method[0] in ["nli", "nli_with_query" ] and len(scoring_method) == 1:
                model_name = "deberta_nli"
            elif scoring_method[0] in ["bertscore", "rarebertscore"] and len(scoring_method) == 1:
                model_name = "deberta"
            else:
                model_name = None

            
        else:
            assert False, "Now not support mix 'zh' and 'en'"
        

        scores_dict = defaultdict(lambda : [[] for _ in range(len(language_list))])

        flatten_index_list = [0]
        flatten_cands = []
        flatten_refs = []
        flatten_queries = []
        
        for query_index, responses in enumerate(responses_list):
            if type(responses) is str or len(responses) == 1:
                #scores_list[query_index] = [1.0]
                for method in scoring_method:
                    scores_dict[method][query_index] = torch.tensor([1.0], dtype=torch.double)
                flatten_index_list.append(flatten_index_list[-1])
                continue
            elif len(responses) == 0:
                flatten_index_list.append(flatten_index_list[-1])
                continue
            
            for i in range(len(responses)):
                for j in range(len(responses)):
                    if i == j:
                        continue
                    flatten_cands.append(responses[i])
                    flatten_refs.append(responses[j])
                    flatten_queries.append(queries[query_index])
            flatten_index_list.append(flatten_index_list[-1] + len(responses) * (len(responses) - 1))
        
        

        all_preds = self.scoring_responses(scoring_method, flatten_queries, flatten_cands, flatten_refs, device=device,  batch_size=self.batch_size, model_name=model_name) if len(self.result_dict) == 0 else self.scoring_responses_from_cache(scoring_method, flatten_queries, flatten_cands, flatten_refs, device=device,  batch_size=self.batch_size, model_name=model_name)


        for query_index, responses in enumerate(responses_list):
            begin = flatten_index_list[query_index]
            end = flatten_index_list[query_index + 1]
            
            if begin == end:
                assert type(responses) is str or len(responses) == 1 or len(responses) == 0
                continue
            # 
            # scores = [0.0 for _ in range(len(responses))]
            # for i in range(len(responses)):
                # for j in range(len(responses) - 1):
                    # if all_preds[begin + i * (len(responses) - 1) + j][2] > scores[i]:
                        # scores[i] = all_preds[begin + i * (len(responses) - 1) + j][2].item()
            for method in scoring_method:
                
                scores_dict[method][query_index] = all_preds[method][begin:end].view(len(responses), len(responses)-1)
        

        
        for method in scores_dict.keys():
            if method in self.bidirectional:
                if method == "em":
                    gathering_method = "max"
                else:
                    gathering_method = "mean"

                scores_dict[method] = [gather_bidirectional_nli_score(s_matrix, gathering_method=gathering_method) for s_matrix in scores_dict[method]]
            else:
                continue

        
        return scores_dict
    
    def batch_voting(self, queries, language_list, responses_list, device="gpu0", voting_weight=None, knowledge_names = None):
        """
        calculate the voting score.
        Parameters:
        queries - List[str]: questions
        language_list - List[str]: list of 'en' or 'zh',  identify the language used.
        responses_list - List[List[str]] or List[List[List[str]]]: the sampled responses of the corresponding query. m*n*(k), m is the query num, n is the corresponding knowledge num, the third dimision k is optional which is the num of sampled responses for each query response pair
        device: str - 'cpu' or f'gpu{rank},  identify the device used to calculate score, 

        return:
        List[List[double]] - queries_num * responses_num - the voting score of each query-response pair
        """
        

        if voting_weight is not None:
            assert len(voting_weight) == len(responses_list[0]), f"voting_weight should be with the same size as knowledge_num"
            voting_weight = torch.tensor(voting_weight, dtype=torch.float64)
            # weight = voting_weight/voting_weight.mean()
            weight = voting_weight
        else:
            weight = torch.ones(len(responses_list[0]),dtype=torch.float64)
        

        if isinstance(responses_list[0][0], list):
            if self.hierarchical_voting:
                expand_queries = [queries[q_id] for q_id in range(len(queries)) for _ in responses_list[q_id]]
                expand_language_list = [language_list[q_id] for q_id in range(len(queries)) for _ in responses_list[q_id]]
                expand_responses_list = [r for q_id in range(len(queries)) for r in responses_list[q_id]]
                expand_scores_matrix_dict = self._batch_voting(expand_queries, expand_language_list, expand_responses_list, device=device)
                expand_scores_matrix = []
                for query_index in range(len(expand_queries)):
                    expand_matrix_list = [self.composition_weight[k]*expand_scores_matrix_dict[k][query_index] for k in self.scoring_method]
                    expand_scores_matrix.append(reduce(lambda x,y: x+y, expand_matrix_list))

                expand_partial_flatten = partial(pooling_score, pooling_method=self.h_voting["pooling_method"], threshold=self.h_voting["pooling_threshold"], min_acceptance_num=self.h_voting["min_acceptance_num"], topk = self.h_voting["topk"])
                scores_list = list(map(expand_partial_flatten, expand_scores_matrix))
                filter_identifier = [(s >= self.threshold).int() for s in scores_list] 
                expand_index_list = [torch.argmax((score*iden)).item() for score, iden in zip(scores_list, filter_identifier)]
                flatten_responses_list = [r[i] for i, r in zip(expand_index_list, expand_responses_list)]
                start_id = 0
                _responses_list = []
                for rs in responses_list:
                    end_id = start_id + len(rs)
                    _responses_list.append(flatten_responses_list[start_id: end_id])
                    start_id = end_id
                responses_list = _responses_list
            else:
                knowledge_names = [k+f"_{j+1}" for i, k in enumerate(knowledge_names) for j in range(len(responses_list[0][i]))]
                weight = torch.tensor([w for i, w in enumerate(weight.tolist()) for _ in range(len(responses_list[0][i]))], dtype=torch.float64)
                responses_list = [list(chain(*r)) for r in responses_list]
        
        if len(responses_list[0]) == 1:
            if self.verbose:
                logger.info("only one response candidate for each queries, no need to vote")
            
            return [[1.0] for _ in responses_list], responses_list, knowledge_names
        
                



        scores_matrix_dict = self._batch_voting(queries, language_list, responses_list, device=device)
        scores_matrix = []
        for query_index in range(len(queries)):
            matrix_list = [self.composition_weight[k]*scores_matrix_dict[k][query_index] for k in self.scoring_method]
            scores_matrix.append(reduce(lambda x,y: x+y, matrix_list))


        partial_flatten = partial(pooling_score, pooling_method=self.pooling_method, threshold=self.pooling_threshold, min_acceptance_num=self.min_acceptance_num, topk = self.topk)
        scores_list = list(map(partial_flatten, scores_matrix))
        filter_identifier = [(s >= self.threshold).int() for s in scores_list] 
        
        self.info_dict["config"] = self.config
        self.info_dict["scores"] = dict()

        for query, responses, scores, idens in zip(queries, responses_list, scores_list, filter_identifier):
            
            self.info_dict["scores"][query] = []
            for response, score, iden in zip(responses, scores.tolist(), idens.tolist()):
                self.info_dict["scores"][query].append({"response":response, "score": score, "keeped": bool(iden) })
                

        return [(score*iden*weight).tolist() for score, iden in zip(scores_list, filter_identifier)], responses_list, knowledge_names
    
    def scoring_responses(self, scoring_methods, queries, hyps, refs, device="cpu", batch_size = 64, model_name=None):
        """
        scoring_methods: str or List[str] - methods used to score the similarity of sentence pair, should be subset of ["nli", "bertscore", "rarebertscore", "em", "f1]
        hyps: List[str] - list of hypothesis
        refs: List[str] - list of references
        device: str - "cpu" or f"gpu{id}",
        batch_size: int
        model_name: str - model used for calculating score
        """
        
        if isinstance(scoring_methods, str):
            scoring_methods = [scoring_methods]
        
        for m in scoring_methods:
            assert m in METHODS_LIST, f"method should be subset of [{'|'.join(METHODS_LIST)}], but '{m}' is given "

        scores = dict()

        if "nli" in scoring_methods:
            if model_name is None:
                nli_model_name = EN_NLI_MODEL_NAME
            else:
                nli_model_name = model_name
            scores["nli"]  = NLIScore(nli_model_name, hyps, refs, device, batch_size=batch_size)
                
        if "bertscore" in scoring_methods:
            if model_name is None:
                bs_model_name = EN_BERT_SCORE_MODEL_NAME
            else:
                bs_model_name = model_name

            scores["bertscore"] = BertScore(bs_model_name, hyps, refs, device, batch_size=batch_size)[:, 2]
        if "rarebertscore" in scoring_methods:
            if model_name is None:
                rbs_model_name = EN_BERT_SCORE_MODEL_NAME
            else:
                rbs_model_name = model_name
            scores["rarebertscore"] = RareBertScore(rbs_model_name, hyps, refs, device,batch_size=batch_size, Corpus=None, top_p=0.5)[:, 2]
        if "em" in scoring_methods:
            scores["em"]= EMScore(hyps, refs)
        if "f1" in scoring_methods:
            scores["f1"] = F1Score(hyps, refs)

        if "nli_with_query" in  scoring_methods:
            if model_name is None:
                nli_q_model_name = EN_NLI_MODEL_NAME
            else:
                nli_q_model_name = model_name

            scores["nli_with_query"]  = NLIScore(nli_q_model_name, hyps, refs, device, batch_size=batch_size, queries=queries, with_query=True)


        

        for method in scores.keys():
            for q, h,r, s in zip(queries, hyps, refs, scores[method].tolist()):
                
                
                self.result_dict[self.key(q,h,r,method)] = s
                self.renewed_result_dict[self.key(q,h,r,method)] = s

        return scores

    def scoring_responses_from_cache(self, scoring_methods, queries, hyps, refs, device="cpu", batch_size = 64, model_name=None):
        cache = set(self.result_dict.keys())
        scores = dict()
        
        for method in scoring_methods:
            no_cache_queries = []
            no_cache_hyps = []
            no_cache_refs = []
            for query, hyp, ref in zip(queries, hyps, refs):
                if self.key(query, hyp, ref, method) not in cache:
                    no_cache_queries.append(query)
                    no_cache_hyps.append(hyp)
                    no_cache_refs.append(ref)
            if len(no_cache_queries) > 0:
                if self.verbose:
                    logger.info(f"{len(no_cache_queries)} non cached hyp-ref pairs are found for method [{method}], start to calculate from scratch")
                self.scoring_responses(method, no_cache_queries, no_cache_hyps, no_cache_refs, device=device, batch_size=batch_size, model_name=model_name)
            else:
                if self.verbose:
                    logger.info(f"all hyp-ref pairs are found for method [{method}].")
        
        for method in scoring_methods:
            scores[method] = torch.tensor([self.result_dict[self.key(query, hyp, ref, method)] for query, hyp, ref in zip(queries, hyps, refs)], dtype=torch.double)
            for query, hyp, ref in zip(queries, hyps, refs):
                self.renewed_result_dict[self.key(query, hyp, ref, method)] = self.result_dict[self.key(query, hyp, ref, method)]
        
        return scores
