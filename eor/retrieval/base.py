import json

from loguru import logger
from tqdm import tqdm


class BaseRetriever:
    def __init__(self, config):
        self.result_dict = {}
        self.info_dict = {}
                

        self.log_detail = config["verbose"] if config["verbose"] is not None else True       
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]
        
        if config["load_result_path"] is not None:
            if self.log_detail:
                logger.warning(f"[Load docs from {config['load_result_path']}]")

            self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None :
            if self.log_detail:
                logger.warning(f"[Load info from {config['load_info_path']}]")
            self.load_info(config["load_info_path"])

    def retrieve(self, query, language, device="gpu0"):
        return self.result_dict[query] if query in self.result_dict.keys() else self._retrieve(query, language, device)
    
    def batch_retrieve(self, queries, language_list, device="gpu0", world_size=1):
        return self._batch_retrieve(queries, language_list, device) if len(self.result_dict) == 0 else self._batch_retrieve_use_cache(queries, language_list, device, world_size=world_size)
    
    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))

    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))
    
    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
    
    def _retrieve(self, query, language, device="gpu0"):
        pass

    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        if not self.log_detail:
            return [self._retrieve(query, language, device) for query, language in zip(queries, language_list)]
        else:
            return [self._retrieve(queries[index], language_list[index], device) for index in tqdm(range(len(queries)))]
    
    def _batch_retrieve_use_cache(self, queries, language_list, device="gpu0", world_size=1):
        cache = set(self.result_dict.keys())
        no_cache_queries = [query for query in queries if query not in cache]
        no_cache_language_list = [language for query, language in zip(queries, language_list) if query not in cache]
        if len(no_cache_queries) > 0:
            self._batch_retrieve(no_cache_queries, no_cache_language_list, device)
        return [self.result_dict[query] for query in queries]