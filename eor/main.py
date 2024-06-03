import json
import math
import os

import torch.distributed as dist
import time
from loguru import logger
import datetime
from itertools import chain

from eor.data import load_example
from eor.response import StandardGenerator
from eor.voting import StandardVoter
from eor.scorer import RewardScorer, aggregate_voting_rewarding_scores
from eor.controller import ControllerConstructor
from eor.evaluation import Evaluator 
from .utils import LLM, collect_saved_files


def run_eor(rank, config, debug=False):
    # llm
    LLM.get_llm_config(config["LLMConfig"])
    LLM.gpu_ids = config["device"]
    LLM.ddp = config["ddp"]
    verbose = config["log_detail"]

    
    if rank is None:
        world_size = 1
        rank = 0 if config["device"][0] == "cpu" else config["device"][0]
    else:
        world_size = len(config["device"])
    
    if config["device"][0] == "cpu":
        device = "cpu"
    else:
        device = f"gpu{rank}"
    
    if config["ddp"]:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.barrier()

        if rank == 0 and verbose:
            logger.info(f"********************DDP connection finished for all rank********************")

    
    # load data
    logger.info(f"********************{device}: start to load dataset from {config['test_file']}********************")
    begin_time = time.time()
    dataset = load_example(config["test_file"])
    # dataset = dataset.shuffle(seed=42)

    
    if config["ddp"]:
        part_num = math.ceil(len(dataset)/world_size)
        begin, end = part_num * rank, part_num * (rank + 1)

        queries = dataset["query"][begin:end]
        language_list = dataset["language"][begin:end]
        truthful_answer_list = dataset["truthful answer"][begin:end]
        
        dist.barrier()
        
    else:
        queries = dataset["query"]
        language_list = dataset["language"]
        truthful_answer_list = dataset["truthful answer"]
    
    if rank == 0 or not config["ddp"]:

        logger.info(f"********************Dataset Loaded for all ranks, total batch size: {len(dataset)}, batch size for each rank: {len(queries)}********************")
    
    
    load_time_usage = time.time() - begin_time
    
    # replace rank placeholder in save_path
    
    if config["result_path"] is not None and "$rank$" in config["result_path"]:
        config["result_path"] = config["result_path"].replace("$rank$", str(rank))
    
    if config["result_info_path"] is not None and "$rank$" in config["result_info_path"]:
        config["result_info_path"] = config["result_info_path"].replace("$rank$", str(rank))
    
    for name, module in config.active_modules.items():

        
        if module["save_result_path"] is not None and "$rank$" in module["save_result_path"]:
            module["save_result_path"] = module["save_result_path"].replace("$rank$", str(rank))
            
        if module["save_info_path"] is not None and "$rank$" in module["save_info_path"]:
            module["save_info_path"] = module["save_info_path"].replace("$rank$", str(rank))
    
    # controller, take the responsiblity to retrieve and process all documents
    
    logger.info(f"{device}: save result to {config['output_dir'].replace('$rank$/', '')}")
    logger.info(f"********************{device}: start to retrieve and process knowledge with controller********************")
    
    begin_time = time.time()

    Controller = ControllerConstructor(config, device, world_size=world_size, rank=rank)
    processed_knowledge = Controller.batch_process(queries, language_list)
    Controller.save_if_set()
     
    if config["ddp"]:
        dist.barrier()
    
    controller_time_usage = time.time() - begin_time
        
    if rank == 0 or not config["ddp"]:
        logger.info(f"********************knowledge processing finished for all rank, time used: {datetime.timedelta(seconds = controller_time_usage)}********************")
        if config["ddp"]:
            collect_saved_files(config["output_dir"], world_size)
    
    if config["ddp"]:
        dist.barrier()
        
    # start to generate responses
    
    if config.generator is not None:
        
        generator = StandardGenerator(config["ModuleConfig"][config.generator])

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
            logger.info(f"********************{device}: start to generate retrieval augmented responses")
            begin_time = time.time()
            _responses_list = generator.batch_response(_query_list, _language_list, _knowledge_list, device, world_size=world_size)
            generator.save_if_set(new_only=True)
            logger.info(f"********************{device}: knowledge groundred responses generation finished, time used {datetime.timedelta(seconds = (time.time() - begin_time))}")
        else:
            begin_time = time.time()
            knowledge_num = 0
            

            
        
        if config["GeneratorConfig"]["dynamic_retrieval"]:
            logger.info(f"********************{device}: start to generate query only responses")
            generator.set_query_only_mode()
            q_only_begin_time = time.time()
            query_only_responses = generator.batch_response(queries, language_list, None, device, world_size=world_size)
            
            _language_list.extend(language_list)
            knowledge_names.append("query_only")
            _knowledge_list.extend([None]*knowledge_num)
            _responses_list.extend(query_only_responses)
            logger.info(f"********************{device}: query only responses generation finished, time used {datetime.timedelta(seconds = (time.time() - q_only_begin_time))}")
        
        response_time_usage_for_rank = time.time() - begin_time
        logger.info(f"********************{device}: responses generation finished, time used {datetime.timedelta(seconds = response_time_usage_for_rank)}")
        generator.save_if_set(new_only=True)

        responses_list = []

        for query_index in range(len(queries)):
            
            responses_list.append({k_name: _responses_list[k_index * len(queries) + query_index] for k_index, k_name in enumerate(knowledge_names)})
        
        response_to_save = {q: a for q,a in zip(queries, responses_list)}
        
        logger.info(f"save generator results to {os.path.join(config['output_dir'].replace('$rank$', str(rank)), 'generator_result.json')}")
        json.dump(response_to_save, open(os.path.join(config["output_dir"].replace("$rank$", str(rank)), "generator_result.json"), "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
       
        if config["ddp"]:
            dist.barrier()
         
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************responses generation finished for all rank********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)

    response_time_usage = time.time() - begin_time

    if config["ddp"]:
        
        dist.barrier()
        

    LLM.release_all()
    begin_time = time.time()
    
    # vote
    if config.voter is not None:
        logger.info(f"********************{device}:Start to Voting")
        
        voter = StandardVoter(config["ModuleConfig"]["Voter"])
        if voter.use_method_weight:
            
            logger.info("method weight is used for voting!!")
            weight = [config.method_weight[k] for k in knowledge_names]
            logger.info(f"method weight used for each method is: {json.dumps({k: w for k, w in zip(knowledge_names, weight)}, indent=4)}")
        else:
            logger.info("method weight is Not used for voting, each method is treated as same.")
            weight = None
        _responses_list = [[responses_list[query_idx][name] for name in knowledge_names] for query_idx in range(len(responses_list))]
        _vote_scores_list, _responses_list, knowledge_names  = voter.batch_voting(queries, language_list, _responses_list, device, knowledge_names=knowledge_names, voting_weight=weight) 
        voter.save_if_set(new_only=True)
        responses_list = []

        for query_index in range(len(queries)):
            responses_list.append({k_name: _responses_list[query_index][k_index] for k_index, k_name in enumerate(knowledge_names)})

        vote_scores_list = [{name:vote_score for name, vote_score in zip(knowledge_names, _vote_scores_list[i])} for i in range(len(queries))]    
        
        if config["ddp"]:
            dist.barrier()
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************Voting finished for all rank, time used {datetime.timedelta(seconds = time.time() - begin_time)}********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)
    else:
        logger.info(f"********************{device}: Voter is closed, directly go to Scorer")
        vote_scores_list = [{name: 1.0 for name in knowledge_names} for i in range(len(queries))]
    
    vote_time_usage = time.time() - begin_time

    
    # score
    begin_time = time.time()
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
        logger.info(f"********************{device}: Start to Scoring")
        
        scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
        scores_list = scorer.batch_score(select_query_list, select_language_list, select_response_list, device)
        scorer.save_if_set()

        if config["ddp"]:
            dist.barrier()
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************Scoring finished for all rank, time used {datetime.timedelta(seconds = time.time() - begin_time)}********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)
    else:
        scores_list = list(chain(*select_voting_score_list))

    
    score_time_usage = time.time() - begin_time
    
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
    
    time_usage_dict = {
        "load_time_usage": load_time_usage,
        "controller_time_usage": controller_time_usage,
        "response_time_usage": response_time_usage,
        "vote_time_usage": vote_time_usage,
        "score_time_usage": score_time_usage
    }
    
    logger.info(time_usage_dict.__str__())
    if config["result_info_path"] is not None:
        json.dump(time_usage_dict, open(config["result_info_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent =4)

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

    if config["result_path"] is not None:
        json.dump(result, open(config["result_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        logger.info(f"********************{device}: result file saved to {config['result_path']}********************")

    if config["ddp"]:
        dist.barrier()

    if rank == 0 or not config["ddp"]:
        logger.info(f"********************WHOLE PROCESS FINISHED!!!********************")
        if config["ddp"]:
            collect_saved_files(config["output_dir"], world_size)
    
    if rank == 0 or not config["ddp"]:
        if config["run_evaluation"]:
            evaluator = Evaluator("NQ", metrics=["em","bem"])

            score_column_mapping = {"em": "em_score", "bem": "bem_score"}
            cached_score_path = os.path.join(config["cached_score_path"], "results.xlsx") if config["cached_score_path"] is not None else None
            
            evaluator.evaluate(result, cached_score_path=cached_score_path, cached_score_name_mapping=score_column_mapping, system_only=False, verbose=verbose)
            cached_dir = os.path.normpath(config["output_dir"]).replace("$rank$", "")
            evaluator.result.to_excel(cached_dir)
            em_score = evaluator.result.scores["em_score"]["system_answer"]
            bem_score = evaluator.result.scores["bem_acc"]["system_answer"]
            logger.info("*"*10 + f"[SYSTEM FINAL SCORE] em: {em_score} | bem {bem_score} on dataset [{config['data']} - {config['test_file'].split('/')[-1][:-6]}] "+"*"*10)
    
