import json

from loguru import logger
import torch


class BaseScorer:
    def __init__(self, config):
        self.result_dict = {}
        self.info_dict = {}
        
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]
        
        if config["load_result_path"] is not None:
            logger.warning(f"[Load docs from {config['load_result_path']}]")
            self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None:
            logger.warning(f"[Load info from {config['load_info_path']}]")
            self.load_info(config["load_info_path"])
            
    def score(self, query, language, response, gpu="gpu0"):
        return self.result_dict[self.key(query, response)] if self.key(query, response) in self.result_dict.keys() else self._score(query, language, response, gpu)
    
    def batch_score(self, queries, language_list, response_list, gpu="gpu0"):
        return self._batch_score(queries, language_list, response_list, gpu) if len(self.result_dict) == 0 else self._batch_score_use_cache(queries, language_list, response_list, gpu)
    
    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))

    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))
    
    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    def _score(self, query, language, response, gpu="gpu0"):
        return self._batch_score([query], [language], [response], gpu=gpu)

    def _batch_score(self, queries, language_list, response_list, gpu="gpu0"):
        pass
    
    @staticmethod
    def key(query, response):
        key = f"query: {query}\n\nresponse: {response}"
        return key

    def _batch_score_use_cache(self, queries, language_list, response_list, gpu="gpu0"):
        cache = set(self.result_dict.keys())
        no_cache_queries = []
        no_cache_language_list = []
        no_cache_response_list = []
        for query, language, response in zip(queries, language_list, response_list):
            if self.key(query, response) not in cache:
                no_cache_queries.append(query)
                no_cache_language_list.append(language)
                no_cache_response_list.append(response)
        if len(no_cache_queries) > 0:
            self._batch_score(no_cache_queries, no_cache_language_list, no_cache_response_list, gpu)
        return [self.result_dict[self.key(query, response)] for query, response in zip(queries, response_list)]
    

def aggregate_voting_rewarding_scores(voting_scores, reward_scores, ignore_voting=False):
    """
    voting_scores: List[float]
    reward_scores: List[float]
    """
    if ignore_voting:
        voting_weight = 0.0
    else:
        voting_weight = 0.5
    reward_weight = 1 - voting_weight
    scores = voting_weight * torch.tensor(voting_scores, dtype=torch.double) + reward_weight * torch.tensor(reward_scores, dtype=torch.double)


    return scores.tolist()