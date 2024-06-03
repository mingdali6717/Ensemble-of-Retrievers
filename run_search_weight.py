from eor.parameter_search import *
import argparse
import os
import numpy as np
import math

from functools import partial
from eor.data import load_example
from eor.evaluation import Evaluator
from scipy.optimize import minimize, shgo

# POOLING_METHOD = "majority_voting"
POOLING_METHOD = "mean" # method used for pooling
HEURISTIC_THRESHOLD = 0.0
POOLING_THRESHOLD = 0.8



HEURISTIC_SCORING_WEIGHT = [0.0, 1.0, 0.0, 0.0] # intial weight for nli; bertscore; em; nli with query respectively
HEURISTIC_METHOD_LIST = ["c_all", "gc",  "gc_w", "gc_w_sum", "gendoc", "gg", "gm", "gm_w", "gm_w_sum", "query_only","wiki"]  #the weight of the heuristic method will be used as start point for the search. initial weight not in heuristic_method_list will be set to zero
#COMPLETE_METHODS_COMBINATION_LIST = ["query_only", "gendoc", "wiki", "gc", "gc_w", "c_all", "gm_w", "gm", "gg"]
COMPLETE_METHODS_COMBINATION_LIST = ["c_all", "c_all_sum","gc", "gc_sum", "gc_w", "gc_w_sum", "gendoc", "gg", "gg_sum", "gm", "gm_sum", "gm_w", "gm_w_sum", "query_only","wiki", "wiki_sum"]  # all active methods for weight search


OUTPUT_DIR = "run_parameter_search_result"

#tq-dev-em: [0.45, 0. ,0.52, 0.27, 0.60, 0.56, 0.56, 0.59,  0.02 ] 0.7493
#tq-dev shgo [0.83668305 0.2115232  0.53581684 0.72578824 0.97066432 0.82069426 0.96635126 0.91440567 0.25871614]
THRESHOLD_BOUND = [(0,0.3)]
SCORE_WEIGHT_BOUND = [(0.0,1), (0,1), (0.01,1), (0,1)]
METHOD_WEIGHT_BOUND = [[0.0, 0.6] for _ in range(len(COMPLETE_METHODS_COMBINATION_LIST))]


def run_nelder_mead_weight_search(x, config=None, queries=None, language_list=None, truthful_answer_list=None, processed_knowledge=None, responses_list=None, cached_score=None, knowledge_names=None, voter_result_dict=None, methods_weight=None, methods_combination_list=None, threshold=None, scoring_weight=None, dev_info=None, test_info=None, file_to_write=None, search_by="em", print_detail_params=False):
        x_param_name = []
        if threshold == None:
            threshold = x[0]
            x_param_name += ["threshold"]
            base_index = 1
        else:
            base_index = 0
        
        if scoring_weight is None:
            x_param_name += ["nli", "bertscore", "em", "nli/w/q"]
            composition_weight = [x[base_index], x[base_index+1], 0.0, x[base_index+2], 0.0, x[base_index+3]]
            base_index += 4
        else:
            composition_weight = [scoring_weight[0], scoring_weight[1], 0.0, scoring_weight[2], 0.0, scoring_weight[3]]
        
        if methods_weight is None:
            x_param_name += methods_combination_list
            assert len(x[base_index:]) == len(methods_combination_list), "something is wrong"
            methods_weight = {m:w for w, m in zip(x[base_index:], methods_combination_list)}
        if print_detail_params:
            x_param_string = " ".join([f"{n}: {float(v):.5f}" for n, v in zip(x_param_name, x)])
        else:
            x_param_string = " ".join([f"{n}: {float(v):.2f}" for n, v in zip(x_param_name, x)])
        print(f"{x_param_string}")
        
        params = {
        "scoring_method": "composition",
        "threshold": threshold,
        "pooling_method": POOLING_METHOD,
        "composition_weight": composition_weight,
        "pooling_threshold": POOLING_THRESHOLD,
        "mean_pooling_topk": -1,
        "min_acceptance_num": -1,
        }

        result = run_one_param_config(config, queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, voter_result_dict, params, methods_weight)
        evaluator = Evaluator("NQ", metrics=["em","bem"])
        evaluator.evaluate(result, cached_score=cached_score, cached_score_name_mapping=score_column_mapping, system_only=True, verbose=verbose)
        em_score = evaluator.result.scores["em_score"]["system_answer"]
        bem_score = evaluator.result.scores["bem_score"]["system_answer"]
        score_string = f"train_score: [EM]{em_score:.4f} [BEM]{bem_score:.4f}"
        score_to_write = f"[TRAINEM]{em_score:.4f}[TRAINBEM]{bem_score:.4f}"

        if dev_info is not None:
            dev_info["params"] = params
            dev_info["weight"] = methods_weight
            dev_result = run_one_param_config(**dev_info)
            dev_evaluator = Evaluator("NQ", metrics=["em","bem"])
            
            dev_evaluator.evaluate(dev_result, cached_score=dev_info["cached_score"], cached_score_name_mapping=score_column_mapping, system_only=True, verbose=verbose)
            dev_em_score = dev_evaluator.result.scores["em_score"]["system_answer"]
            dev_bem_score = dev_evaluator.result.scores["bem_score"]["system_answer"]
            score_string = score_string + f", dev_score: [EM]{dev_em_score:.4f} [BEM]{dev_bem_score:.4f}"
            score_to_write = score_to_write + f"[DEVEM]{dev_em_score:.4f}[DEVBEM]{dev_bem_score:.4f}"

        if test_info is not None:
            test_info["params"] = params
            test_info["weight"] = methods_weight
            test_result = run_one_param_config(**test_info)
            test_evaluator = Evaluator("NQ", metrics=["em","bem"])
            
            test_evaluator.evaluate(test_result, cached_score=test_info["cached_score"], cached_score_name_mapping=score_column_mapping, system_only=True, verbose=verbose)
            test_em_score = test_evaluator.result.scores["em_score"]["system_answer"]
            test_bem_score = test_evaluator.result.scores["bem_score"]["system_answer"]
            score_string = score_string + f", test_score: [EM]{test_em_score:.4f} [BEM]{test_bem_score:.4f}"
            score_to_write = score_to_write + f"[TESTEM]{test_em_score:.4f}[TESTBEM]{test_bem_score:.4f}"
        
        score_string += "\n"
        score_to_write += "\n"
        print(score_string)
        if file_to_write is not None:
            file_to_write.write(f"{x_param_string}{score_to_write}\n")
            
                
        if search_by == "em":
            return -em_score
        elif search_by == "bem":
            return -bem_score
        elif search_by == "mean":
            return -(em_score + bem_score)/2
        
    

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='a project on retrieval enhance with controller')
    parser.add_argument('-c', '--config', type=str, default='config/base.yaml', help='config file(yaml) path')
    parser.add_argument('-o', '--output_dir', type=str, help='output_dir_to_save_results')
    parser.add_argument("-w", "--write_to_file",  type=str, help="file to write search intermediate result")
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='name of dataset to evaluate, should be one of nq, wq or tq')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='name of the model to search form, only support llama-7b, llama-13b, turbo')
    parser.add_argument("-v", "--verbose", action='store_true')

    parser.add_argument('-t', '--train_path', type=str, required=True, help='path of saved query for train data')
    parser.add_argument('-r', '--cached_path', type=str, required=True, help='path of cached result dir')
    parser.add_argument("--eval_at_last", action="store_true", help="if true, will only evaluate the test set at the final step")
    parser.add_argument('--dev_path', type=str, help='path of saved query for test data')
    parser.add_argument('--dev_cached_path', type=str, help='path of cached result dir')
    parser.add_argument('--test_path', type=str, help='path of saved query for test data')
    parser.add_argument('--test_cached_path', type=str, help='path of cached result dir')

    parser.add_argument('--split_train_dev', action="store_true", help='if true, split the train set into train/dev with ratio 9:1')

    
    parser.add_argument("-cn", "--controller_result_name", type=str, help="file name of the cached controller result file")
    parser.add_argument("-gn", "--generator_result_name", type=str, help="file name of the cached generator result file")

    parser.add_argument("--search_threshold", action="store_true", help="if True, search for the threshold")
    parser.add_argument("--scoring_weight_only", action="store_true", help="if True, will use 0.0 for threshold , bertscore for threshold, only searh for the method weight")
    parser.add_argument("--method_weight_only", action="store_true", help="if True, will used, only searh for the threshold and scoring_weight")
    parser.add_argument("--optimization_method", default="nelder-mead", type=str, help="the optimization method used")
    parser.add_argument("--max_methods_num", default=-1, type=int, help="max num of methods used to search, denote by k, then top-k best performed methods will be used")
    parser.add_argument("-u", "--method_weight_upper_bound", type=float, default=0.6, help="the upper bound for all method weight")
    parser.add_argument("--train_percentage", type=float, default=1.0, help="the percentage of trainset used to search params")
    parser.add_argument("--seed", type=int, default=42, help="the seed used to sample train set, and split train, dev set")

    parser.add_argument("--search_by", choices=["em", "bem", "mean"], default="em", help="metrics used to calculate objective function")
    
    
    args, _ = parser.parse_known_args()
    assert args.method_weight_upper_bound > 0.1, f"the method_weight_upper_bound should be bigger than 0.1, but {args.method_weight_upper_bound} is given.  all methods with weight below than 0.1 will be ignored. "
    verbose = args.verbose
    if args.eval_at_last:
        assert (getattr(args, "dev_path", None) is not None) and (getattr(args, "dev_cached_path", None) is not None), "eval_at_last is set to be true, dev_path and dev_cached_path should be given."


    if getattr(args, "output_dir", None) is not None:
        output_dir = args.output_dir
    else:
        output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    # set default weight and methods list
    DEFAULT_WEIGHT = HEURISTIC_WEIGHT[args.model_name][args.dataset_name]
    COMPLETE_METHODS_COMBINATION_LIST = sorted(COMPLETE_METHODS_COMBINATION_LIST, key=lambda x:DEFAULT_WEIGHT[x], reverse=True)
    if args.max_methods_num > 2 and args.max_methods_num <= len(COMPLETE_METHODS_COMBINATION_LIST):
        print(f"MAX_METHODS_NUM IS GIVEN, ONLY TOP {args.max_methods_num} will be used to search")
        COMPLETE_METHODS_COMBINATION_LIST = COMPLETE_METHODS_COMBINATION_LIST[:args.max_methods_num]
        METHOD_WEIGHT_BOUND = [[0.0, 0.6] for _ in range(len(COMPLETE_METHODS_COMBINATION_LIST))]
    
    # set up the file to write search process
    if getattr(args, "write_to_file", None) is not None:
        if not args.write_to_file.endswith(".txt"):
            file_path = args.write_to_file + ".txt"
        else:
            file_path = args.write_to_file
        file_to_write = open(os.path.join(output_dir, file_path), "a")
    else:
        name = args.dataset_name + "-" + args.model_name + "-" + args.optimization_method
        if args.search_threshold:
            name += "-" + "search_threshold"
        if args.method_weight_only:
            name += "-" + "method_weight_only"
        elif args.scoring_weight_only:
            name += "-" + "scoring_weight_only"
        else:
            name += "-" + "all"
        
        name += "-" + POOLING_METHOD + "-" + str(HEURISTIC_THRESHOLD) +"-" + f"methods_num_{len(COMPLETE_METHODS_COMBINATION_LIST)}" + "-" + f"upperbound_{args.method_weight_upper_bound}" + "-" + f"trainpercent_{args.train_percentage}"

        file_to_write= open(os.path.join(output_dir, name+".txt"), "a")
    
    
    # load cached intermediate result
    default_config = load_default_config(args.config, args.cached_path, args.train_path)
    controller_file_name = getattr(args, "controller_result_name", None)
    generator_file_name = getattr(args, "generator_result_name", None)
    controller_result_dict, generator_result_dict, voter_result_dict = get_result_dict(args.cached_path,controller_file_name=controller_file_name, generator_file_name=generator_file_name)
    
    cached_score_path = os.path.join(args.cached_path, "results.xlsx")
    score_column_mapping = {"em": "em_score", "bem": "bem_score"}
    base_evaluator = Evaluator("NQ", metrics=["em","bem"])
    cached_score = base_evaluator.load_cached_score(cached_score_path, score_column_mapping)
    original_dataset = load_example(default_config["test_file"])
    dataset_size = math.ceil(args.train_percentage * original_dataset.shape[0])
    dataset = original_dataset.shuffle(seed=args.seed).select(range(dataset_size))

    dev_info = None

    # prepare the cached result for dev set
    if getattr(args, "dev_path", None) is not None:
        assert getattr(args, "dev_cached_path", None) is not None, "cached path should be given for the dev set"
        print("dev path is given, dev set performance will be shown at each search step")
        dev_default_config = load_default_config(args.config, args.dev_cached_path, args.dev_path)
        dev_controller_result_dict, dev_generator_result_dict, dev_voter_result_dict = get_result_dict(args.dev_cached_path,controller_file_name=controller_file_name, generator_file_name=generator_file_name)

        dev_cached_score_path = os.path.join(args.dev_cached_path, "results.xlsx")
        dev_cached_score = base_evaluator.load_cached_score(dev_cached_score_path, score_column_mapping)

        dev_dataset = load_example(dev_default_config["test_file"])
        dev_info = {
            "dataset": dev_dataset,
            "controller_result_dict": dev_controller_result_dict,
            "generator_result_dict": dev_generator_result_dict,
            "voter_result_dict": dev_voter_result_dict,
            "cached_score": dev_cached_score
        }
    
    elif args.split_train_dev:
        dev_default_config = default_config
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        dataset = split_dataset["train"]
        dev_dataset = split_dataset["test"]
        dev_default_config = default_config
        dev_controller_result_dict, dev_generator_result_dict, dev_voter_result_dict = (controller_result_dict, generator_result_dict, voter_result_dict)
        dev_cached_score = cached_score
        dev_info = {
            "voter_result_dict": dev_voter_result_dict,
            "cached_score": dev_cached_score
        }

    test_info = None

    # prepare the cached result for test set
    if getattr(args, "test_path", None) is not None:
        assert getattr(args, "test_cached_path", None) is not None, "cached path should be given for the test set"
        print("test path is given, test set performance will be shown at each search step")
        test_default_config = load_default_config(args.config, args.test_cached_path, args.test_path)
        test_controller_result_dict, test_generator_result_dict, test_voter_result_dict = get_result_dict(args.test_cached_path,controller_file_name=controller_file_name, generator_file_name=generator_file_name)

        test_cached_score_path = os.path.join(args.test_cached_path, "results.xlsx")
        test_cached_score = base_evaluator.load_cached_score(test_cached_score_path, score_column_mapping)

        test_dataset = load_example(test_default_config["test_file"])
        test_info = {
            "dataset": test_dataset,
            "controller_result_dict": test_controller_result_dict,
            "generator_result_dict": test_generator_result_dict,
            "voter_result_dict": test_voter_result_dict,
            "cached_score": test_cached_score
        }

    
    # set up the weight upper bound for searching
    default_method_weight = {m:0.0 for m in COMPLETE_METHODS_COMBINATION_LIST}
    for i, m in enumerate(COMPLETE_METHODS_COMBINATION_LIST):         
        METHOD_WEIGHT_BOUND[i][1] = args.method_weight_upper_bound
        if m == "gc_w":
            METHOD_WEIGHT_BOUND[i][0] = 0.11
        
        METHOD_WEIGHT_BOUND[i] = tuple(METHOD_WEIGHT_BOUND[i])

    # set up the initial weight for searching
    for m in HEURISTIC_METHOD_LIST:
        if m not in COMPLETE_METHODS_COMBINATION_LIST:
            continue
        # default_method_weight[m] = DEFAULT_WEIGHT[m] * args.method_weight_upper_bound
        default_method_weight[m] = DEFAULT_WEIGHT[m]

    default_method_weight_list = [default_method_weight[m] for m in COMPLETE_METHODS_COMBINATION_LIST]

    # set the search simplx for nelder nead
    if args.search_threshold:
        bound = THRESHOLD_BOUND
        initial_point = [0.0]
        threshold = None
    else:
        bound = []
        initial_point = []
        threshold = HEURISTIC_THRESHOLD
    
    if args.scoring_weight_only:
        assert args.method_weight_only == False, "only one of 'scoring_weight_only' and 'method_weight_only' can be true"
        bound = bound + SCORE_WEIGHT_BOUND
        initial_point = initial_point + HEURISTIC_SCORING_WEIGHT
        method_weight = default_method_weight
        method_combination_list =  None
        scoring_weight = None
        if args.search_threshold:
            initial_simplex = np.array([[0.0, 1, 0,0,0],
                                    [0.0, 0,1,0,0],
                                    [0.0, 0,0,1,0],
                                    [0.0,0,0,0,1],
                                    [0.1, 0.5,0.5,0.5,0.5],
                                    [0.3, 1, 0,0,0]])
        else:
            initial_simplex = np.array([[ 1, 0,0,0],
                                    [ 0,1,0,0],
                                    [ 0,0,1,0],
                                    [0,0,0,1],
                                    [0.5,0.5,0.5,0.5]])
    elif args.method_weight_only:
        assert args.scoring_weight_only == False, "only one of 'scoring_weight_only' and 'method_weight_only' can be true"
        bound = bound + METHOD_WEIGHT_BOUND
        initial_point = initial_point + [default_method_weight[m] for m in COMPLETE_METHODS_COMBINATION_LIST]
        method_weight=None
        method_combination_list = COMPLETE_METHODS_COMBINATION_LIST 
        scoring_weight = HEURISTIC_SCORING_WEIGHT

        base_simplex =  np.array([default_method_weight_list,]*len(COMPLETE_METHODS_COMBINATION_LIST))
        # base_simplex =  np.array([[0.0 for _ in default_method_weight_list],]*len(COMPLETE_METHODS_COMBINATION_LIST))
        
        for i in range(len(base_simplex)):
            if base_simplex[i][i] > 0:
                base_simplex[i][i] = 0
            else:
                base_simplex[i][i] = 1
        additional_line = np.expand_dims(np.array(default_method_weight_list),axis=0)
        base_simplex = np.concatenate([base_simplex, additional_line], axis=0)

        if args.search_threshold:
            threshold_simplex = np.array([[THRESHOLD_BOUND[0][0]],] * (len(COMPLETE_METHODS_COMBINATION_LIST)+1))
            initial_simplex = np.concatenate([threshold_simplex, base_simplex], axis=1)
            additional_line = np.expand_dims(np.array([THRESHOLD_BOUND[0][1]] + default_method_weight_list),axis=0)
            initial_simplex = np.concatenate([initial_simplex, additional_line], axis=0)
        else:
            initial_simplex = base_simplex
    
    else:
        bound = bound + SCORE_WEIGHT_BOUND + METHOD_WEIGHT_BOUND
        initial_point = initial_point + HEURISTIC_SCORING_WEIGHT + default_method_weight_list
        method_weight=None
        scoring_weight = None
        method_combination_list = COMPLETE_METHODS_COMBINATION_LIST
        s_simplex = [[ 1, 0,0,0],
                        [ 0,1,0,0],
                        [ 0,0,1,0],
                        [0,0,0,1],
                        [0.5,0.5,0.5,0.5]]
        simplex_upper = np.array([s +  default_method_weight_list for s in s_simplex])

        simplex_lower = np.array([default_method_weight_list,]*len(COMPLETE_METHODS_COMBINATION_LIST))
        # simplex_lower =  np.array([[0.0 for _ in default_method_weight_list],]*len(COMPLETE_METHODS_COMBINATION_LIST))

        for i in range(len(simplex_lower)):
            if simplex_lower[i][i] > 0:
                simplex_lower[i][i] = 0
            else:
                simplex_lower[i][i] = 1
        simplex_lower = np.concatenate([np.array([HEURISTIC_SCORING_WEIGHT,]*len(COMPLETE_METHODS_COMBINATION_LIST)), simplex_lower], axis=1)
        initial_simplex = np.concatenate([simplex_upper, simplex_lower], axis=0)
        if args.search_threshold:
            threshold_simplex = np.array([[THRESHOLD_BOUND[0][0]],] * len(initial_simplex))
            initial_simplex = np.concatenate([threshold_simplex, initial_simplex], axis=1)
            additional_line = np.expand_dims(np.array([THRESHOLD_BOUND[0][1]] + HEURISTIC_SCORING_WEIGHT + default_method_weight_list), axis=0)
            initial_simplex = np.concatenate([initial_simplex, additional_line], axis=0)
    

    # prepare the prerequisite results for voting
    queries, language_list, truthful_answer_list, processed_knowledge, responses_list, knowledge_names, config = prepare_for_voting(default_config, COMPLETE_METHODS_COMBINATION_LIST, dataset, controller_result_dict, generator_result_dict, verbose=verbose)

    if dev_info is not None:

        dev_queries, dev_language_list, dev_truthful_answer_list, dev_processed_knowledge, dev_responses_list, dev_knowledge_names, dev_config = prepare_for_voting(dev_default_config, COMPLETE_METHODS_COMBINATION_LIST, dev_dataset, dev_controller_result_dict, dev_generator_result_dict, verbose=verbose)
        dev_info["queries"] = dev_queries
        dev_info["language_list"] = dev_language_list
        dev_info["truthful_answer_list"] = dev_truthful_answer_list
        dev_info["processed_knowledge"] = dev_processed_knowledge
        dev_info["responses_list"] = dev_responses_list
        dev_info["knowledge_names"] = dev_knowledge_names
        dev_info["config"] = dev_config
    
    if test_info is not None:

        test_queries, test_language_list, test_truthful_answer_list, test_processed_knowledge, test_responses_list, test_knowledge_names, test_config = prepare_for_voting(test_default_config, COMPLETE_METHODS_COMBINATION_LIST, test_dataset, test_controller_result_dict, test_generator_result_dict, verbose=verbose)
        test_info["queries"] = test_queries
        test_info["language_list"] = test_language_list
        test_info["truthful_answer_list"] = test_truthful_answer_list
        test_info["processed_knowledge"] = test_processed_knowledge
        test_info["responses_list"] = test_responses_list
        test_info["knowledge_names"] = test_knowledge_names
        test_info["config"] = test_config

    if args.optimization_method == "nelder-mead":
        print("*"*10 + f"START OPTIMIZATION WITH NELDER MEAD METHOD, SEARCH BY {args.search_by}" + "*"*10 )
        print("*"*10 + "BOUND" + "*"*10 + f"\n{bound}")
        print("*"*10 + "INITIAL POINT" + "*"*10 + f"\n{initial_point}")
        print("*"*10 + "SIMPLEX" + "*"*10 + f"\n{initial_simplex}")

        if args.eval_at_last:
            test_info_during_search = None
        else:
            test_info_during_search = test_info
   
        search_func = partial(run_nelder_mead_weight_search, config=config, queries=queries, language_list=language_list, truthful_answer_list=truthful_answer_list, processed_knowledge=processed_knowledge, responses_list=responses_list, knowledge_names=knowledge_names, methods_weight = method_weight, scoring_weight = scoring_weight, threshold=threshold, methods_combination_list = COMPLETE_METHODS_COMBINATION_LIST, voter_result_dict=voter_result_dict, cached_score=cached_score, dev_info=dev_info, test_info=test_info_during_search, file_to_write = file_to_write, search_by=args.search_by)

        optimization_result = minimize(search_func, initial_point, method="Nelder-Mead", bounds = bound, options = {"disp":True, "return_all": True, "initial_simplex": initial_simplex})
            
        print(f"x: {optimization_result.x}, success: {optimization_result.success}, messages: {optimization_result.message}\ntrain path{args.train_path}")
        file_to_write.write(f"[FinalResult] search with [{args.search_by}]\n")
        
        run_nelder_mead_weight_search(optimization_result.x, config=config, queries=queries, language_list=language_list, truthful_answer_list=truthful_answer_list, processed_knowledge=processed_knowledge, responses_list=responses_list, knowledge_names=knowledge_names, methods_weight = method_weight, scoring_weight = scoring_weight, threshold=threshold, methods_combination_list = COMPLETE_METHODS_COMBINATION_LIST, voter_result_dict=voter_result_dict, cached_score=cached_score, dev_info=dev_info, test_info=test_info, file_to_write = file_to_write, print_detail_params=True)

            # initial_point = (0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
            # 
# 
            # initial_simplex = np.array([[0.0, 1, 0,0,0, 1, 1, 1, 1, 1, 1],
                                        # [0.0, 0,1,0,0, 1, 1, 1, 1, 1, 1],
                                        # [0.0, 0,0,1,0, 1, 1, 1, 1, 1, 1],
                                        # [0.0,0,0,0,1, 1, 1, 1, 1, 1, 1],
                                        # [0.1, 0.5,0.5,0.5,0.5, 1, 1, 1, 1, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 1, 1, 1, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 0.5, 1, 1, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 1, 0.5, 1, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 1, 1, 0.5, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 0.5, 1, 1, 1, 1, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 1, 1, 1, 0.5, 1],
                                        # [0.1, 0, 0.5,0,0.5, 1, 1, 1, 1, 1, 0.5]])

    elif args.optimization_method == "shgo":
        print("*"*10 + f"START OPTIMIZATION WITH SHGO, SEARCH BY {args.search_by}" + "*"*10 )
        print("*"*10 + "BOUND" + "*"*10 + f"\n{bound}")

        if args.eval_at_last:
            test_info_during_search = None
        else:
            test_info_during_search = test_info
       
        search_func = partial(run_nelder_mead_weight_search, config=config, queries=queries, language_list=language_list, truthful_answer_list=truthful_answer_list, processed_knowledge=processed_knowledge, responses_list=responses_list, knowledge_names=knowledge_names, methods_weight = method_weight, scoring_weight = scoring_weight, threshold=threshold, methods_combination_list = COMPLETE_METHODS_COMBINATION_LIST, voter_result_dict=voter_result_dict, cached_score=cached_score, dev_info=dev_info, test_info=test_info_during_search,file_to_write = file_to_write, search_by=args.search_by)
        
        optimization_result = shgo(search_func, bound,  sampling_method = "simplicial", options = {"disp":True})
        
        print(f"x: {optimization_result.x}, success: {optimization_result.success}, messages: {optimization_result.message}\ntrain path: {args.train_path} pooling_method: {POOLING_METHOD} pooling_threshold: {POOLING_THRESHOLD}")

        
        file_to_write.write(f"[FinalResult] search with [{args.search_by}]\n")
        run_nelder_mead_weight_search(optimization_result.x, config=config, queries=queries, language_list=language_list, truthful_answer_list=truthful_answer_list, processed_knowledge=processed_knowledge, responses_list=responses_list, knowledge_names=knowledge_names, methods_weight = method_weight, scoring_weight = scoring_weight, threshold=threshold, methods_combination_list = COMPLETE_METHODS_COMBINATION_LIST, voter_result_dict=voter_result_dict, cached_score=cached_score, dev_info=dev_info, test_info=test_info, file_to_write = file_to_write, print_detail_params=True)
    
    else:
        raise KeyError(f"optimization method should be one of nelder-mead or shgo, but {args.optimization_method} is give")
    
    if file_to_write is not None:
        file_to_write.close()
    




