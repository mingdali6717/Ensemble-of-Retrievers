import json
import os
import time
import copy

import dynamic_yaml
from collections import defaultdict
from itertools import chain
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm
from ..controller import METHOD_MODULE_MAPPING

WQ_DEFAULT_WEIGHT = {"c_all":0.602834665,
"c_all_sum":0.549783276,
"gc":0.603317881,
"gc_sum":0.557985577,
"gc_w":0.60935737,
"gc_w_sum":0.561693931,
"gendoc":0.550378486,
"gg":0.564828793,
"gg_sum":0.5277845,
"gm":0.592946156,
"gm_sum":0.54799379,
"gm_w":0.610902674,
"gm_w_sum":0.541743047,
"query_only":0.565378789,
"wiki":0.616722245,
"wiki_sum":0.554145865}

NQ_DEFAULT_WEIGHT = {"c_all": 0.576433,
"c_all_sum": 0.552392,
"gc": 0.624559,
"gc_sum": 0.589006,
"gc_w": 0.628043,
"gc_w_sum": 0.594947,
"gendoc": 0.4316  ,
"gg": 0.570783,
"gg_sum": 0.534789,
"gm": 0.611518,
"gm_sum": 0.567084,
"gm_w" : 0.62286 ,
"gm_w_sum": 0.562967,
"query_only": 0.422837,
"wiki": 0.569146,
"wiki_sum": 0.524521}

TQ_DEFAULT_WEIGHT= {
"c_all": 0.749130433,
"c_all_sum": 0.729848766,
"gc": 0.795997772,
"gc_sum": 0.77102996,
"gc_w": 0.80866645,
"gc_w_sum": 0.76852698,
"gendoc": 0.612197532,
"gg": 0.764767231,
"gg_sum": 0.737413083,
"gm": 0.791511335,
"gm_sum": 0.755196323,
"gm_w": 0.798845202,
"gm_w_sum": 0.742413721,
"query_only": 0.623642895,
"wiki": 0.701507151,
"wiki_sum": 0.661401904}

class Config:
    """Configurator module that load the defined parameters."""

    def __init__(self, config_file, debug=False, config_dict=None):
        """

        Load parameters and set log level.

        Args:
            config_file (str): path to the config file, which should be in ``yaml`` format.
                You can use default config provided in the `Github repo`_, or write it by yourself.
            debug (bool, optional): whether to enable debug function during running. Defaults to False.

        """
        if config_dict is not None:
            default_config = config_dict
        else:
            default_config = self.load_yaml_configs(config_file)
        
        default_config["time"] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # set log
            
        log_name = default_config.get("log_name", default_config["time"]) + ".log"
        if not os.path.exists("log"):
            os.makedirs("log")
        logger.remove()
        if debug:
            level = 'DEBUG'
        else:
            level = 'INFO'
        logger.add(os.path.join("log", log_name), level=level)
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)
        
        # set default values for weight of retriever methods and voting methods
        ignore_method_weight = 0.1 if default_config["ignore_method_weight"] is None else default_config["ignore_method_weight"]

        if default_config["cache_parameter_dir"] is not None:
            param_dict = self.read_optimal_parameters(default_config["cache_parameter_dir"])
            
            logger.info("cached parameter setting found, reset config by the given paramter setting!!!")
            logger.info(f"cached scoring weight:\n{json.dumps(param_dict['voting_weight'], indent=4)}")
            logger.info(f"cached method weight:\n{json.dumps(param_dict['method_weight'], indent=4)}\nmethod with weight <= {ignore_method_weight} will be ignored!!!")
            
            self.voter_composition_weight = [param_dict["voting_weight"]["nli"], param_dict["voting_weight"]["bertscore"], 0.0, param_dict["voting_weight"]["em"], 0.0, param_dict["voting_weight"]["nli/w/q"]]
            active_methods_by_param = [m for m,v in param_dict["method_weight"].items() if v > ignore_method_weight]
            self.method_weight = param_dict["method_weight"]
            logger.info(f"method with weight <= {ignore_method_weight} will be ignored!!!\nactive method weight:\n{json.dumps({m: self.method_weight[m] for m in active_methods_by_param},indent=4)}")
            self.opt = reset_config(default_config, active_methods_by_param)
             
        else:
            if default_config["data"] == "open_natural_question":
                self.method_weight = NQ_DEFAULT_WEIGHT
            elif default_config["data"] == "webqa":
                self.method_weight = WQ_DEFAULT_WEIGHT
            elif default_config["data"] == "triviaqa":
                self.method_weight = TQ_DEFAULT_WEIGHT
            else:
                raise ValueError("something is wrong")

            self.voter_composition_weight = None
            self.opt = default_config
            
        # set gpu and ddp
        if not torch.cuda.is_available():
            gpu = False
            if self.opt["log_detail"]:
                logger.warning("[WARNING]: GPU IS NOT AVAILABLE!!!! RUN WITH CPU.")
            self.opt["ddp"] = False
            self.opt["device"] = ["cpu"]
            gpu_num = 0
        else:
            gpu = self.opt["gpu"]
        
            if gpu is None:
                gpu = [i for i in range(torch.cuda.device_count())]
            elif type(gpu) is int:
                gpu = [gpu]
            elif type(gpu) is str:
                gpu = [int(i) for i in range(len(gpu.replace(" ", "").split(',')))]
            
            gpu_num = len(gpu)

            if gpu_num == 1:
                self.opt["ddp"] = False
            else:
                self.opt["ddp"] = True
            self.opt["device"] = gpu
            logger.info(f"RUN WITH {gpu_num} GPU. GPU IDs ARE: {gpu}")
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu)
        
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        
        # set retriever methods and processing modules
        self.available_modules = self.get_available_modules()
        
        self.active_methods = {name: method for name, method in self.opt["ControlConfig"].items() if method["turn_on"]}

        self.default_modules = self.check_default_modules(METHOD_MODULE_MAPPING)
        
        
        self.active_modules = list()

        # check args
        self.check_config()

        self.opt["ActiveMethods"] = self.active_methods
        self.opt["ActiveModules"] = self.active_modules

        if self.opt["output_dir"] is not None:
            config_path = os.path.normpath(self.opt["output_dir"].replace("$rank$", ""))
            json.dump(self.opt, open(os.path.join(config_path, "config.json"), 'w', encoding="UTF-8"), indent=4)

        # set seed

        self.set_seed(self.opt["seed"], gpu_num)
        if self.opt["log_detail"]:
            logger.info(f"[ACTIVE METHODS]:{len(self.active_methods.keys())} methods are activated. method names are as follows:\n" + json.dumps(self.active_methods, indent=4))
            logger.info(f"[ACTIVE MODULES]:{len(self.active_modules.keys())} modules are activated. module names are as follows:\n" + json.dumps(self.active_modules, indent=4))
        else:
            logger.info(f"[ACTIVE METHODS]:{len(self.active_methods.keys())} methods are activated. method names are as follows:{'|'.join(self.active_methods.keys())}")
            logger.info(f"[ACTIVE MODULES]:{len(self.active_modules.keys())} modules are activated. module names are as follows :{'|'.join(self.active_modules.keys())}")
        # if self.opt["log_detail"]:
            # logger.info("[Config]:" + '\n' + json.dumps(self.opt, indent=4))
       

    @staticmethod
    def load_yaml_configs(filename):
        """This function reads ``yaml`` file to build config dictionary

        Args:
            filename (str): path to ``yaml`` config

        Returns:
            dict: config

        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(dynamic_yaml.dump(dynamic_yaml.load(f.read()))))
        return config_dict

    @staticmethod
    def set_seed(seed: int, gpu_num: int):
        """
        Set seed for numpy and torch
        :param seed: int
        :param gpu_num: num of gpu to use
        :return: None
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if gpu_num > 0:
            torch.cuda.manual_seed_all(seed)

    def check_config(self):
        if self.opt["test_file"] is not None:
            assert os.path.isfile(self.opt["test_file"]), f"{self.opt['test_file']} doesn't exists."
        
        # makedir for output_dir
        if self.opt["output_dir"] is not None:
            self.opt["output_dir"] = os.path.join(self.opt["output_dir"],self.opt["time"]) 
            
            if self.opt["ddp"]:
                self.opt["output_dir"] = os.path.join(self.opt["output_dir"], "$rank$/")
            
            if "$rank$" in self.opt["output_dir"]:
                for rank in range(len(self.opt["device"])):
                    _dir = self.opt["output_dir"].replace("$rank$", str(rank))
                    if not os.path.exists(_dir):
                        os.makedirs(_dir, exist_ok=False)
            else:     
                if not os.path.exists(self.opt["output_dir"]):
                    os.makedirs(self.opt["output_dir"], exist_ok=False)   
        

        # get output_path
        if self.opt["output_dir"] is not None and self.opt["result_file"] is not None:
            self.opt["result_path"] = os.path.join(self.opt["output_dir"], self.opt["result_file"])
        else:
            self.opt["result_path"] = None
        
        if self.opt["output_dir"] is not None and self.opt["result_info_file"] is not None:
            self.opt["result_info_path"] = os.path.join(self.opt["output_dir"], self.opt["result_info_file"])
        else:
            self.opt["result_info_path"] = None

        # check ResponseConfig Module:
        
        
        if self.opt["GeneratorConfig"]["module"] is not None:
            self.check_custom_validility(self.opt["GeneratorConfig"]["module"], "Generator")
        else:

            self.opt["GeneratorConfig"]["module"] = "Generator"
        
        if self.opt["GeneratorConfig"]["without_retrieval"] and self.opt["GeneratorConfig"]["turn_on"]:
            self.active_methods = dict()
            self.opt["GeneratorConfig"]["dynamic_retrieval"] = True
            
        if self.opt["GeneratorConfig"]["dynamic_retrieval"] == True:
            self.opt["ModuleConfig"][self.opt["GeneratorConfig"]["module"]]["dynamic_retrieval"] = True
        else:
            self.opt["ModuleConfig"][self.opt["GeneratorConfig"]["module"]]["dynamic_retrieval"] = False
        
        if self.opt["GeneratorConfig"]["turn_on"]:
            self.active_modules.append(self.opt["GeneratorConfig"]["module"])
            self.generator = self.opt["GeneratorConfig"]["module"]
        else:
            self.generator = None
        

        # check EvaluatorConfig Module:
        if self.opt["EvaluatorConfig"]["VoterModule"]["module"] is not None:
            self.check_custom_validility(self.opt["EvaluatorConfig"]["VoterModule"]["module"], "Voter")
        else:

            self.opt["EvaluatorConfig"]["VoterModule"]["module"] = "Voter"
        
        if self.opt["EvaluatorConfig"]["VoterModule"]["turn_on"]:
            if self.generator is None:
                raise KeyError("generator is turned off but voter is turned on!!!")
            self.active_modules.append(self.opt["EvaluatorConfig"]["VoterModule"]["module"])
            self.voter = self.opt["EvaluatorConfig"]["VoterModule"]["module"]

            if "num_responses_per_prompt" in self.opt["ModuleConfig"][self.generator]["generate_kwargs"] and self.opt["ModuleConfig"][self.generator]["generate_kwargs"]["num_responses_per_prompt"] > 1:
                if "hierarchical_voting" not in self.opt["ModuleConfig"][self.voter] :
                    self.opt["ModuleConfig"][self.voter]["hierarchical_voting"] = {"turn_on": False, "pooling_method": "majority_voting", "pooling_threshold": None, "min_acceptance_num": None, "mean_pooling_topk": None}

                else:
                    if self.opt["ModuleConfig"][self.voter]["hierarchical_voting"]["turn_on"] is None:
                        self.opt["ModuleConfig"][self.voter]["hierarchical_voting"]["turn_on"] = False
                    if "pooling_method" not in self.opt["ModuleConfig"][self.voter]["hierarchical_voting"]:
                        self.opt["ModuleConfig"][self.voter]["hierarchical_voting"]["pooling_method"] = "majority_voting"
                    for n in ["pooling_threshold", "min_acceptance_num", "mean_pooling_topk"]:
                        if n not in self.opt["ModuleConfig"][self.voter]["hierarchical_voting"]:
                            self.opt["ModuleConfig"][self.voter]["hierarchical_voting"][n] = None
            else:
                self.opt["ModuleConfig"][self.voter]["hierarchical_voting"] = {"turn_on": False, "pooling_method": "majority_voting", "pooling_threshold": None, "min_acceptance_num": None, "mean_pooling_topk": None}
            
            if self.voter_composition_weight is not None:
                self.opt["ModuleConfig"][self.voter]["scoring_method"] = "composition"
                self.opt["ModuleConfig"][self.voter]["composition_weight"] = self.voter_composition_weight
            
        else:
            self.voter = None

        # check ScorerConfig Module:
        if self.opt["EvaluatorConfig"]["ScorerModule"]["module"] is not None:
            self.check_custom_validility(self.opt["EvaluatorConfig"]["ScorerModule"]["module"], "Scorer")
            
        else:

            self.opt["EvaluatorConfig"]["ScorerModule"]["module"] = "Scorer"
        
        if self.opt["EvaluatorConfig"]["ScorerModule"]["turn_on"]:
            if self.generator is None:
                raise KeyError("generator is turned off but Scorer is turned on!!!")
            self.active_modules.append(self.opt["EvaluatorConfig"]["ScorerModule"]["module"])
            self.scorer = self.opt["EvaluatorConfig"]["ScorerModule"]["module"]
        else:
            self.scorer = None

        
        # check controller 
        active_methods = dict()
        
        for name, method in self.active_methods.items():
            assert method["method_name"] in METHOD_MODULE_MAPPING.keys(), f"method name should be one of supported names: {', '.join(list(METHOD_MODULE_MAPPING.keys()))}. but '{method['method_name']}' is given"
            

            if method["Modules"]["retriever"] is not None:
                retrievers = method["Modules"]["retriever"]
                if isinstance(retrievers, str):
                    retrievers = [retrievers]
                for r_name in retrievers:
                    assert r_name in self.available_modules["Retriever"], f"given retrieverer '{r_name}' for method {name} is not availialbe, only following retrieval modules are suported: {', '.append(self.available_modules['Retriever'])}. "
                    r_type = self.opt["ModuleConfig"][r_name]["type"]
                    default_retrievers ={self.opt["ModuleConfig"][mod]["type"] : mod for mod in self.default_modules[method["method_name"]]["retriever"]}
                    if r_type not in default_retrievers.keys():
                        logger.warning(f"method '{name}' need retrieval methods: {', '.join(list(default_retrievers.keys()))}, but method '{r_name}' with type '{r_type}' is given which will not be used'")
                    else:
                        default_retrievers[r_type] = r_name
                
                method["Modules"]["retriever"] = [n for n in default_retrievers.values()]
            else:
                method["Modules"]["retriever"] = self.default_modules[method["method_name"]]["retriever"]


            if "contriever" not in METHOD_MODULE_MAPPING[method["method_name"]] and method["Modules"]["contriever"] is not None:
                logger.warning(f"method '{method['method_name']}' do not need contriever, but a contriever module'{method['Modules']['contriever']}' is given, which will not be used")
                method["Modules"]["contriever"] = None
            elif "contriever" in METHOD_MODULE_MAPPING[method["method_name"]] and method["Modules"]["contriever"] is not None:
                assert method["Modules"]["contriever"] in self.available_modules["Contriever"], f"given contriever '{method['Modules']['contriever']}' for method {name} is not availialbe, only following contrieval modules are suported: {', '.join(self.available_modules['Contriever'])}. "
                
                assert isinstance(method["Modules"]["contriever"], str), f"only one contriever method for one process method, but more than 1 are given for method '{name}'"
                method["Modules"]["contriever"] = [method["Modules"]["contriever"]]
            
            elif "contriever" in METHOD_MODULE_MAPPING[method["method_name"]] and method["Modules"]["contriever"] is None:
                method["Modules"]["contriever"] = self.default_modules[method["method_name"]]["contriever"]


            if not method["summarize"]:
                if method["Modules"]["summarizer"] is not None:
                    logger.warning(f"Summarization for method '{method['method_name']}' is turned off, but a summarizer '{method['Modules']['summarizer']}' is given, which will not be used")

                    method["Modules"]["summarizer"] = None
                
            else:
                if method["Modules"]["summarizer"] is not None:
                    assert method["Modules"]["summarizer"] in self.available_modules["Summarizer"], f"given summarizer '{method['Modules']['summarizer']}' for method {name} is not availialbe, only following summarization modules are suported: {', '.append(self.available_modules['Summarizer'])}. "

                    assert isinstance(method["Modules"]["summarizer"], str), f"only one summarizer module is permitted for one process method, but more than 1 are given for method '{name}'"
                    method["Modules"]["summarizer"] = [method["Modules"]["summarizer"]]
                else:
                    method["Modules"]["summarizer"] = self.default_modules[method["method_name"]]["summarizer"]
            
            
            
            if not method["summarize"]:
                active_methods[name] = {
                    "method_name": method["method_name"],
                    "full_name": method["method_name"] + "-" + "-".join(sorted(chain(*[v for v in method["Modules"].values() if v is not None])))
                }
                active_methods[name].update(method["Modules"])
            else:
                active_methods[name+"_sum"] = {
                    "method_name": method["method_name"],
                    "full_name": method["method_name"] + "-" + "-".join(sorted(chain(*[v for v in method["Modules"].values() if v is not None])))
                }
                active_methods[name+"_sum"].update(method["Modules"])
                if method["keep_both"]:
                    method["Modules"]["summarizer"] = None
                    active_methods[name] = {
                    "method_name": method["method_name"],
                    "full_name": method["method_name"] + "-" + "-".join(sorted(chain(*[v for v in method["Modules"].values() if v is not None])))
                }
                    active_methods[name].update(method["Modules"])
        self.active_methods = active_methods

        self.retrieval_modules = set(chain(*[v["retriever"] for v in self.active_methods.values() if v["retriever"] is not None]))
        self.contriever_modules = set(chain(*[v["contriever"] for v in self.active_methods.values() if v["contriever"] is not None]))
        self.summarizer_modules = set(chain(*[v["summarizer"] for v in self.active_methods.values() if v["summarizer"] is not None]))

        self.controller_modules = set.union(self.retrieval_modules, self.contriever_modules, self.summarizer_modules)

        self.active_modules.extend(list(self.controller_modules))
        

        self.active_modules = {name: self.opt["ModuleConfig"][name] for name in self.active_modules}

        # check controller config
        if self.opt["cache_dir"] is not None:
            controller_cache_path = os.path.join(self.opt["cache_dir"], "controller_result.json")
            if os.path.isfile(controller_cache_path):
                logger.info(f"find cached controller file, try to load result from '{controller_cache_path}'")
                cached_active_methods = json.load(open(controller_cache_path, "r", encoding="UTF-8"))["active_methods"]
                if cached_active_methods == self.active_methods:
                    self.load_controller_from_cache = True
                else:
                    logger.info(f"found cached controller config is not same as the given config, rebuild controller from scratch.")
                    self.load_controller_from_cache = False
            else:
                self.load_controller_from_cache = False

        for name, module in self.active_modules.items():
            
            if self.opt["cache_dir"] is not None and module["load_result_file"] is not None:
                module["load_result_path"] = os.path.join(self.opt["cache_dir"], module["load_result_file"])
                if not self.load_controller_from_cache or name not in self.controller_modules:
                    assert os.path.isfile(module["load_result_path"]), f"[{name}]: load result path '{module['load_result_path']}' doesn't exists."
            else:
                module["load_result_path"] = None
            
            if self.opt["cache_dir"] is not None and module["load_info_file"] is not None:
                module["load_info_path"] = os.path.join(self.opt["cache_dir"], module["load_info_file"])
                if not self.load_controller_from_cache or name not in self.controller_modules:
                    assert os.path.isfile(module["load_info_path"]), f"[{name}]: load info path '{module['load_info_path']}' doesn't exists."
            else:
                module["load_info_path"] = None
            
            if self.opt["output_dir"] is not None and module["save_result_file"] is not None:
                module["save_result_path"] = os.path.join(self.opt["output_dir"], module["save_result_file"])
            else:
                module["save_result_path"] = None
            
            if self.opt["output_dir"] is not None and module["save_info_file"] is not None:                             
                module["save_info_path"] = os.path.join(self.opt["output_dir"], module["save_info_file"])
            else:
                module["save_info_path"] = None
# 
    def get_available_modules(self):
        available_modules = defaultdict(list)

        for name, module in self.opt["ModuleConfig"].items():
            if module["type"] in ["Web", "Wiki", "Gendoc"]:
                available_modules["Retriever"].append(name)
            elif module["type"] == "Summarizer":
                available_modules["Summarizer"].append(name)
            elif module["type"] == "Contriever":
                available_modules["Contriever"].append(name)
            elif module["type"] == "Generator":
                available_modules["Generator"].append(name)
            elif module["type"] == "Voter":
                available_modules["Voter"].append(name)
            elif module["type"] == "Scorer":
                available_modules["Scorer"].append(name)
            else:
                raise AttributeError(f"module type should be one of : Web, Wiki, Gendoc, Summarizer, Contriever, Generator, Voter, Scorer. but '{module['type']}' is given in module '{name}")
        return available_modules
    

    def check_custom_validility(self, custom_module_name, module_type):
        if module_type in ["Web", "Wiki", "Gendoc"]:
            block = "Retriever"
        else:
            block = module_type
        assert isinstance(custom_module_name, str), f"only one response module is needed, but multiple are for module type {module_type}"
        assert custom_module_name in self.available_modules[block], f"given module name '{custom_module_name}' for module type '{module_type}' is not supported. Supported {module_type} type module names are: {'|'.join(self.available_modules[block])}"
        assert self.opt["ModuleConfig"][custom_module_name]["type"] == module_type, f"given module name '{custom_module_name}' is not a '{module_type}' module"
    
    def check_default_modules(self, default_modules):
        defaults_setting = self.opt["ControllerConfig"]
        new_default = default_modules
        if defaults_setting["Web"] is not None:
            self.check_custom_validility(defaults_setting["Web"], "Web")
            for k, v in new_default.items():
                if "Web" in v["retriever"]:
                    new_default[k]["retriever"] = [m if m != "Web" else defaults_setting["Web"] for m in new_default[k]["retriever"]]
        
        if defaults_setting["Wiki"] is not None:
            self.check_custom_validility(defaults_setting["Wiki"], "Wiki")
            for k, v in new_default.items():
                if "Wiki" in v["retriever"]:
                    new_default[k]["retriever"] = [m if m != "Wiki" else defaults_setting["Wiki"] for m in new_default[k]["retriever"]]

        if defaults_setting["Gendoc"] is not None:
            self.check_custom_validility(defaults_setting["Gendoc"], "Gendoc")
            for k, v in new_default.items():
                if "Gendoc" in v["retriever"]:
                    new_default[k]["retriever"] = [m if m != "Gendoc" else defaults_setting["Gendoc"] for m in new_default[k]["retriever"]]

        if defaults_setting["Contriever"] is not None:
            self.check_custom_validility(defaults_setting["Contriever"], "Contriever")
            for k, v in new_default.items():
                if "contriever" in v:
                    new_default[k]["contriever"] = [defaults_setting["Contriever"]]
        
        if defaults_setting["Summarizer"] is not None:
            self.check_custom_validility(defaults_setting["Summarizer"], "Summarizer")
            for k, v in new_default.items():
                if "summarizer" in v:
                    new_default[k]["summarizer"] = [defaults_setting["Summarizer"]]
        return new_default

    def read_optimal_parameters(self, paramter_path):
        with open(paramter_path, "r", encoding="utf-8") as f:
            param_string = f.readline()
            param_list = param_string.split(" ")
            method = [param_list[i] for i in range(0, len(param_list), 2)]
            param_score = [param_list[i] for i in range(1, len(param_list), 2)]
            param_dict = {
                "voting_weight": {m.replace(":", ""): float(s) for m, s in zip(method[:4], param_score[:4])},
                "method_weight": {m.replace(":", ""): float(s) for m, s in zip(method[4:], param_score[4:])}
                }
            return param_dict
        




    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.opt[key] = value

    def __getitem__(self, item):
        if item in self.opt:
            return self.opt[item]
        else:
            return None

    def get(self, item, default=None):
        """Get value of corresponding item in config

        Args:
            item (str): key to query in config
            default (optional): default value for item if not found in config. Defaults to None.

        Returns:
            value of corresponding item in config

        """
        if item in self.opt:
            return self.opt[item]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.opt

    def __str__(self):
        return str(self.opt)

    def __repr__(self):
        return self.__str__()

def turn_off_all_method(config):
    config = copy.deepcopy(config)
    for k in config["ControlConfig"].keys():
        config["ControlConfig"][k]["turn_on"] = False
        config["ControlConfig"][k]["summarize"] = False
        config["ControlConfig"][k]["keep_both"] = False
    
    config["GeneratorConfig"]["dynamic_retrieval"] = False
    return config

def reset_config(config, active_methods):
    old_config = turn_off_all_method(config)
    if isinstance(active_methods, str):
        active_methods = active_methods.split("&")
    elif not isinstance(active_methods, list):
        print(active_methods)
        raise ValueError("something is wrong!")
    
    
    processed_method = []
    
    for m in active_methods:
        if m == "query_only":
            old_config["GeneratorConfig"]["dynamic_retrieval"] = True
            continue
        if m in processed_method:
            continue

        if m.endswith("sum"):
            m_name = m[:-4]
        else:
            m_name = m
        
        m_sum_name = m+"_sum"

        assert m_name in set(old_config["ControlConfig"].keys()), f"given method name {m_name} is wrong"

        if (m_name in active_methods) and (m_sum_name not in active_methods):
            old_config["ControlConfig"][m_name]["turn_on"] = True
            old_config["ControlConfig"][m_name]["summarize"] = False
            old_config["ControlConfig"][m_name]["keep_both"] = False
        elif (m_name not in active_methods) and (m_sum_name in active_methods):
            old_config["ControlConfig"][m_name]["turn_on"] = True
            old_config["ControlConfig"][m_name]["summarize"] = True
            old_config["ControlConfig"][m_name]["keep_both"] = False
        elif (m_name in active_methods) and (m_sum_name in active_methods):
            old_config["ControlConfig"][m_name]["turn_on"] = True
            old_config["ControlConfig"][m_name]["summarize"] = True
            old_config["ControlConfig"][m_name]["keep_both"] = True
        
        processed_method.append(m_name)
        processed_method.append(m_sum_name)
    
    return old_config
