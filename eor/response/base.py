import json

from loguru import logger
import torch.distributed as dist
import torch
from ..utils import LLM
from ..utils import OPENAI_MODEL_LIST

class BaseGenerator:
    def __init__(self, config):
        self.result_dict = {}
        self.renewed_result_dict = {}
        self.info_dict = {}
        self.query_only = False
        self.verbose = config["verbose"] if config["verbose"] is not None else True
        
        
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]

        if "num_responses_per_prompt" in config["generate_kwargs"] and config["generate_kwargs"]["num_responses_per_prompt"] is not None:
            self.n = config["generate_kwargs"]["num_responses_per_prompt"]
        else:
            self.n = 1

        if config["load_result_path"] is not None :
            if self.verbose:
                logger.warning(f"[Load docs from {config['load_result_path']}]")
            self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None :
            if self.verbose:
                logger.warning(f"[Load info from {config['load_info_path']}]")
            self.load_info(config["load_info_path"])
   
    def response(self, query, language, knowledge, gpu="gpu0"):
        return self.result_dict[self.key(query, knowledge)] if self.key(query, knowledge) in self.result_dict.keys() else self._response(query, language, knowledge, gpu)
    
    def batch_response(self, queries, language_list, knowledge_list, gpu="gpu0", world_size=1):
        return self._batch_response(queries, language_list, knowledge_list, gpu) if len(self.result_dict) == 0 else self._batch_response_use_cache(queries, language_list, knowledge_list, gpu, world_size=world_size)


    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))

    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))

    @staticmethod
    def key(query, knowledge):
        if knowledge is not None:
            return f"query: {query}\nknowledge: {knowledge}\n"
        else: 
            return f"query: {query}\n"

    def save_if_set(self, new_only=False):
        if new_only:
            to_save = self.renewed_result_dict
        else:
            to_save = self.result_dict
        if self.save_result:
            json.dump(to_save, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    def _response(self, query, language, knowledge, gpu="gpu0"):
        pass

    def _batch_response(self, queries, language_list, knowledge_list, gpu="gpu0"):
        pass

    def _batch_response_use_cache(self, queries, language_list, knowledge_list, device="gpu0", world_size=1):
        if self.query_only:
            knowledge_list = [None] * len(queries)
        cache = set(self.result_dict.keys())
        no_cache_queries = []
        no_cache_language_list = []
        no_cache_knowledge_list = []
        for query, language, knowledge in zip(queries, language_list, knowledge_list):
            if self.key(query, knowledge) in cache:
                r = self.result_dict[self.key(query, knowledge)]
                if isinstance(r, list):
                    cached_n = len(r)
                else:
                    cached_n = 1
                if cached_n != self.n:
                    no_cache_queries.append(query)
                    no_cache_language_list.append(language)
                    no_cache_knowledge_list.append(knowledge)
            else:
                no_cache_queries.append(query)
                no_cache_language_list.append(language)
                no_cache_knowledge_list.append(knowledge)
        
        if len(no_cache_queries) > 0 and self.verbose:
            logger.info(f"{device}: {len(no_cache_queries)} non-cached queries are found, start to generate responses from scratch.")
        
        if language_list[0] == "zh":
            model_name = self.config["zh_model_name"]
        else:
            model_name = self.config["en_model_name"]
        
        if model_name not in OPENAI_MODEL_LIST:
            # initialize all model to circumvate dist.barrier() in DistributedDataParralism
        
            if world_size > 1:
                gathered_tensor = torch.zeros(world_size, dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
                if len(no_cache_queries) > 0:
                    have_non_cached = torch.tensor([1], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
                else: 
                    have_non_cached = torch.tensor([0], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))

                dist.all_gather_into_tensor(gathered_tensor, have_non_cached)
                
                if gathered_tensor.sum().item() > 0:
                    
                    LLM.initial_lm(model_name, device, self.verbose)
        
         
        if len(no_cache_queries) > 0:
            
            self._batch_response(no_cache_queries, no_cache_language_list, no_cache_knowledge_list, device)
        results = []
        for query, knowledge in zip(queries, knowledge_list):
            res = self.result_dict[self.key(query, knowledge)]
            self.renewed_result_dict[self.key(query, knowledge)] = res
            results.append(res)
        return results