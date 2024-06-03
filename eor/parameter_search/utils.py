
import warnings
import numpy as np
import pandas as pd

import copy
import json
import math
import os
from itertools import combinations, product


from loguru import logger
from tqdm import tqdm

from itertools import chain
from tabulate import tabulate

from ..response import StandardGenerator
from ..voting import StandardVoter
from ..scorer import RewardScorer, aggregate_voting_rewarding_scores
from ..controller import ControllerConstructor
from ..utils import LLM
from ..evaluation import Evaluator
from ..config import Config, turn_off_all_method, reset_config, NQ_DEFAULT_WEIGHT, WQ_DEFAULT_WEIGHT, TQ_DEFAULT_WEIGHT

warnings.filterwarnings('ignore')



def methods_combination_generator(dataset_name, combination_path=None, max_method_num=6, topn=10):
    methods_combination_list = []
    assert topn>=5, "methods for combination search should be more than 5"
    if combination_path is not None:
        
        with open(combination_path, "r", encoding="utf-8") as f:
            for line in f:
                methods_combination_list.append([m.strip() for m in line.strip().split("&")])
    
    else:
        base_methods = ["query_only", "wiki", "gc_w", "gendoc", "c_all"]
        if dataset_name == 'nq':
            scores = NQ_DEFAULT_WEIGHT
        elif dataset_name == 'wq':
            scores = WQ_DEFAULT_WEIGHT
        elif dataset_name == "tq":
            scores = TQ_DEFAULT_WEIGHT  
        else:
            raise ValueError(f"dataset_name should be one of nq, wq, tq but '{dataset_name}' is given")

        top_methods = [m  for m in sorted(list(scores.keys()), key=lambda x:scores[x], reverse=True) if m not in base_methods]
        
        combination_source = base_methods + top_methods[:topn-5]
        for i in range(2, max_method_num+1, 1):
            methods_combination_list.extend(list(combinations(combination_source, i)))
    
    return methods_combination_list




def run_voting_scoring_composition(config,  queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, rank=0, device="cpu", save_result=False, result_dict=None, weight=None):

    # vote
    if config.voter is not None:
        voter = StandardVoter(config["ModuleConfig"]["Voter"], result_dict=result_dict)
        _responses_list = [[responses_list[query_idx][name] for name in knowledge_names] for query_idx in range(len(responses_list))]
        _vote_scores_list, _responses_list, knowledge_names = voter.batch_voting(queries, language_list, _responses_list, f"gpu{rank}", voting_weight=weight, knowledge_names=knowledge_names) 
        vote_scores_list = [{name:vote_score for name, vote_score in zip(knowledge_names, _vote_scores_list[i])} for i in range(len(queries))]
        responses_list = []

        for query_index in range(len(queries)):
            responses_list.append({k_name: _responses_list[query_index][k_index] for k_index, k_name in enumerate(knowledge_names)})    
         
    else: 
        vote_scores_list = [{name: 1.0 for name in knowledge_names} for i in range(len(queries))]
    # score
    select_query_list, select_response_list, select_language_list, select_name_list, select_voting_score_list = [], [], [], [], []
    select_index = [0]
    answer_list = [None for _ in range(len(queries))]
    answer_name_list = ["" for _ in range(len(queries))]
    name_list = []
    
    for query_index in range(len(queries)):
        vote_scores = vote_scores_list[query_index]
        filter_names = [name for name, score in vote_scores.items() if score > 0.0]
        filter_scores = [vote_scores[name] for name in filter_names]
        if len(filter_names) == 1:            
            answer_list[query_index] = responses_list[query_index][filter_names[0]]
            answer_name_list[query_index] = filter_names[0]
            select_index.append(select_index[-1] + 1)
            select_voting_score_list.append(filter_scores)
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][filter_names[0]])
            select_name_list.append(filter_names[0])
            continue       
        
        if len(filter_names) == 0:
            filter_names = list(vote_scores.keys())
            filter_scores = list(vote_scores.values())

        for name in filter_names:
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][name])
            select_name_list.append(name)
        
        select_index.append(select_index[-1] + len(filter_names))
        select_voting_score_list.append(filter_scores)

    if config.scorer is not None:
        scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
        scores_list = scorer.batch_score(select_query_list, select_language_list, select_response_list, f"gpu{rank}")        
    else:
        scores_list = list(chain(*select_voting_score_list))
 
    LLM.release_all()
    
    # answer
    reward_scores = []
    final_scores = []
    for query_index in range(len(queries)):
        select_responses = select_response_list[select_index[query_index]:select_index[query_index+1]]
        select_reward_scores = scores_list[select_index[query_index]:select_index[query_index+1]]
        select_queries = select_query_list[select_index[query_index]:select_index[query_index+1]]
        select_names = select_name_list[select_index[query_index]:select_index[query_index+1]]
        select_voting_score = select_voting_score_list[query_index]
       
        for i in range(1, len(select_queries)):
            assert select_queries[i] == select_queries[0] 
        
        if config.voter is None:
            ignore_voting = True
        else: 
            ignore_voting = False
        
        select_scores = aggregate_voting_rewarding_scores(select_voting_score, select_reward_scores, ignore_voting=ignore_voting)
        answer_list[query_index] = select_responses[select_scores.index(max(select_scores))]
        answer_name_list[query_index] = select_names[select_scores.index(max(select_scores))]
        name_list.append(select_names)
        reward_scores.append({n: s for n, s in zip(select_names, select_reward_scores)})
        final_scores.append({n: s for n, s in zip(select_names, select_scores)})

    #save result
    result = dict()
    for query_index, query in enumerate(queries):
        result[query] = {
            "system_answer": answer_list[query_index],
            "system_answer_name": answer_name_list[query_index]
        }
        result[query]["ground_truth_answer"] = truthful_answer_list[query_index]
        result[query]["details"] = dict()
        
        for name in knowledge_names:
            knowledge = processed_knowledge[name][query_index] if (name in processed_knowledge and processed_knowledge is not None) else None
            temp = {
                "knowledge": knowledge, 
                "response": responses_list[query_index][name],
                "vote_score": vote_scores_list[query_index][name],
                # 因为奖励模型的评分不是每个回复都有，所以只给出最后答案，不在这里给每个答案的评分。
            }
            if name in name_list[query_index]:
                temp["reward_score"] = reward_scores[query_index][name]
                temp["final_score"] = final_scores[query_index][name]
            else: 
                temp["final_score"] = temp["vote_score"]
            result[query]["details"][name] = temp

    if save_result:
        json.dump(result, open(config["result_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        logger.info(f"********************{device}: result file saved to {config['result_path']}********************")
    return result

def get_result_dict(cached_dir, controller_file_name=None, generator_file_name=None):
    controller_name = controller_file_name if controller_file_name is not None else "controller_result.json"
    generator_name = generator_file_name if generator_file_name is not None else "response_result.json"
    controller_result_dict = json.load(open(os.path.join(cached_dir, controller_name), "r", encoding='UTF-8'))
    generator_result_dict = json.load(open(os.path.join(cached_dir, generator_name), "r", encoding='UTF-8'))
    voter_result_dict = json.load(open(os.path.join(cached_dir, "vote_result.json"), "r", encoding='UTF-8'))
    return controller_result_dict, generator_result_dict, voter_result_dict






def load_default_config(config_path, cached_path, test_path):
    default_config = Config.load_yaml_configs(config_path)
    cached_path = os.path.abspath(cached_path)
    test_file_path = os.path.abspath(test_path)
    default_config['cache_dir'] = cached_path
    default_config['test_file'] = test_file_path
    default_config['output_dir'] = None

    return default_config

def yield_params(scoring_method, pooling_methods, thresholds, pooling_thresholds, knowledge_num, composition_weight=None, mean_pooling_topk=None, min_acceptance_num=None):
    default_params = {
        "scoring_method": None,
        "threshold": None,
        "pooling_method": None,
        "composition_weight": None,
        "pooling_threshold": -1,
        "mean_pooling_topk": -1,
        "min_acceptance_num": -1,
        
    }
    def key(params):
        ke = ""
        for k,v in params.items():
            if isinstance(v, list) or isinstance(v, tuple):
                str_v = [str(i)[:4] for i in v]
                ke += f"[{k}]{'|'.join(str_v)}"
            else:
                ke += f"[{k}]{v}"
        return ke

    pool_size = knowledge_num
    majority_num = math.ceil(pool_size/2)
    if mean_pooling_topk is not None:
        mean_pooling_topk = [k for k in mean_pooling_topk if k>=2 and k<majority_num]
    else:
        mean_pooling_topk = list(range(2, majority_num, 1))
    if min_acceptance_num is not None:
        min_acceptance_num = [k for k in min_acceptance_num if k>=1 and k<majority_num]
    else:
        min_acceptance_num = list(range(1, majority_num, 1))
    
    if composition_weight is not None:
        scoring_method = "composition"
        weight_space = [list(np.arange(*w_range)) if w_range[1] != w_range[0] else [w_range[1]] for w_range in composition_weight]
        
        finished_wight = []
        for c_weight in product(*weight_space):
            if sum(c_weight) == 0:
                continue

            real_weight = [float(w)/sum(c_weight) for w in c_weight]
            
            if real_weight in finished_wight:
                continue
            else:
                finished_wight.append(real_weight)

            for td in thresholds:
                for p_method in pooling_methods:
                    params = copy.deepcopy(default_params)
                    params["scoring_method"] = scoring_method
                    params["pooling_method"] = p_method
                    params["composition_weight"] = c_weight
                    params["threshold"] = td
                    if p_method == "topk":
                        for m_p_k in mean_pooling_topk:
                            params["mean_pooling_topk"] = m_p_k
                            yield key(params), params
                    elif p_method == "voting":
                        for p_threshold in pooling_thresholds:
                            for m_a_n in min_acceptance_num:
                                params["min_acceptance_num"] = m_a_n
                                params["pooling_threshold"] = p_threshold
                                yield key(params), params
                    elif p_method == "majority_voting":
                        for p_threshold in pooling_thresholds:
                            params["pooling_threshold"] = p_threshold
                            yield key(params), params
                    else:
                        yield key(params), params
    else:
        for s_method in scoring_method:
            for td in thresholds:
                for p_method in pooling_methods:
                    params = copy.deepcopy(default_params)
                    params["scoring_method"] = s_method
                    params["pooling_method"] = p_method
                    params["threshold"] = td
                    if p_method == "topk":
                        for m_p_k in mean_pooling_topk:
                            params["mean_pooling_topk"] = m_p_k
                            yield key(params), params
                    elif p_method == "voting":
                        for p_threshold in pooling_thresholds:
                            for m_a_n in min_acceptance_num:
                                params["min_acceptance_num"] = m_a_n
                                params["pooling_threshold"] = p_threshold
                                yield key(params), params
                    elif p_method == "majority_voting":
                        for p_threshold in pooling_thresholds:
                            params["pooling_threshold"] = p_threshold
                            yield key(params), params
                    else:
                        yield key(params), params
                    
                        


def prepare_for_voting(default_config, active_methods, dataset, controller_result_dict, generator_result_dict, verbose=True):
    c = reset_config(default_config, active_methods)
    config = Config(None, config_dict=c)
    config["ModuleConfig"][config.generator]["load_result_file"] = None
    config["ModuleConfig"][config.generator]["load_info_file"] = None
    config["ModuleConfig"]["Voter"]["load_result_file"] = None
    config["ModuleConfig"]["Voter"]["load_result_info"] = None
    

    world_size = 1
    rank = 0 if config["device"][0] == "cpu" else config["device"][0]
    if config["device"][0] == "cpu":
        device = "cpu"
    else:
        device = f"gpu{rank}"
    # load data
    queries = dataset["query"]
    language_list = dataset["language"]
    truthful_answer_list = dataset["truthful answer"]
    # controller and generator
    
    if verbose:
        logger.info(f"********************start to generate responses for voting from cache*************")
    Controller = ControllerConstructor(config, device, world_size=world_size, rank=rank)
    processed_knowledge = Controller.load_from_cache(queries, controller_result_dict, active_methods)
    
    generator = StandardGenerator(config["ModuleConfig"][config.generator])
    generator.result_dict = generator_result_dict
    _query_list = []
    _language_list = []
    knowledge_names = []
    _knowledge_list = []
    _responses_list = []
    if processed_knowledge is not None :
        knowledge_num = len(processed_knowledge.keys())
        _query_list = queries*knowledge_num
        _language_list = language_list*knowledge_num
        for k_name, k_docs in processed_knowledge.items():
            knowledge_names.append(k_name)
            _knowledge_list.extend(k_docs)
        
        _responses_list = generator.batch_response(_query_list, _language_list, _knowledge_list, device, world_size=world_size)
    else:
        knowledge_num = 0
    if config["GeneratorConfig"]["dynamic_retrieval"]:
        
        generator.set_query_only_mode()
        query_only_responses = generator.batch_response(queries, language_list, None, device, world_size=world_size)
        _language_list.extend(language_list)
        knowledge_names.append("query_only")
        _knowledge_list.extend([None]*knowledge_num)
        _responses_list.extend(query_only_responses)
    if verbose:
        logger.info(f"********************responses generation finished*****************")
    responses_list = []
    for query_index in range(len(queries)):
        responses_list.append({k_name: _responses_list[k_index * len(queries) + query_index] for k_index, k_name in enumerate(knowledge_names)})
    LLM.release_all()    

    return queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, config 

def run_one_param_config(config, queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, voter_result_dict, params, weight, **kwargs):

    weight, knowledge_names = tuple(zip(*[(weight[k], k) for k in knowledge_names if weight[k] > 0.1]))

    config["ModuleConfig"]["Voter"].update(params)

    result = run_voting_scoring_composition(config, queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, result_dict=voter_result_dict, weight=weight)

    return result

def run_param_search_for_dataset(dataset, dataset_name, default_config, search_space, cached_score, score_column_mapping, base_output_dir, methods_weight, controller_result_dict, generator_result_dict, voter_result_dict, save_details=False, verbose=False):
    total_result_df = None
    logger.info(f"*******************start to run parameter search for dataset [{dataset_name}], total dataset size: {len(dataset)}********************")

    toy_param_generator = yield_params(search_space["scoring_method"], search_space["pooling_methods"], search_space["thresholds"], search_space["pooling_thresholds"], len(search_space["methods_combination"][0]), composition_weight=search_space["composition_weight"], mean_pooling_topk=search_space["mean_pooling_topk"], min_acceptance_num=search_space["min_acceptance_num"])
    approx_param_space_size = len(["" for _ in toy_param_generator])
    print(f"*"*10 + f"[PARAMETER SPACE SIZE]: {approx_param_space_size}" + "*"*10)

    
    for active_methods in tqdm(search_space["methods_combination"]):
        active_methods = list(active_methods)
        param_generator = yield_params(search_space["scoring_method"], search_space["pooling_methods"], search_space["thresholds"], search_space["pooling_thresholds"], len(active_methods), composition_weight=search_space["composition_weight"], mean_pooling_topk=search_space["mean_pooling_topk"], min_acceptance_num=search_space["min_acceptance_num"])
        name = str(len(active_methods)) + "_" + "&".join(active_methods)
        output_dir = os.path.join(base_output_dir, dataset_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_details:
            detail_dir = os.path.join(output_dir, "results_by_methods")
            if not os.path.exists(detail_dir):
                os.makedirs(detail_dir)
        queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, config = prepare_for_voting(default_config, active_methods, dataset, controller_result_dict, generator_result_dict, verbose=verbose)
        result_df = None
        for k, params in tqdm(param_generator):
            key = name+k
            result = run_one_param_config(config, queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, voter_result_dict, params, methods_weight)
            evaluator = Evaluator("NQ", metrics=["em","bem"])
            evaluator.evaluate(result, cached_score=cached_score, cached_score_name_mapping=score_column_mapping, system_only=True, verbose=verbose)
            score = evaluator.result.scores.rename(index={"system_answer":key})
            if result_df is None:
                result_df = score
            else:
                result_df = pd.concat([result_df, score], axis=0)
            
        if verbose:    
            print(tabulate(result_df, headers='keys', tablefmt='psql'))
        if save_details:
            result_df.to_excel(os.path.join(detail_dir, name+".xlsx"))
        if total_result_df is None:
                total_result_df = result_df
        else:
            total_result_df = pd.concat([total_result_df, result_df], axis=0)
    total_result_df.to_excel(os.path.join(output_dir, dataset_name+"_results.xlsx"))
    return total_result_df
     





