import json

from loguru import logger
from tqdm import tqdm


class BaseConstructor:
    def __init__(self, config):
        self.result_dict = {}
        self.info_dict = {}
        
        self.verbose = config["verbose"] if config["verbose"] is not None else True
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]

        if config["load_result_path"] is not None:
            if self.verbose:
                logger.warning(f"[Load knowledge from {config['load_result_path']}]")
            self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None:
            if self.verbose:
                logger.warning(f"[Load info from {config['load_info_path']}]")
            self.load_info(config["load_info_path"])
        
    def construct(self, query, language, doc, gpu="gpu0"):
        return self.result_dict[self.key(query, doc)] if self.key(query, doc) in self.result_dict.keys() else self._construct(query, language, doc, gpu)
        
    def batch_construct(self, queries, language_list, doc_list, device="gpu0", world_size=1):
        return self._batch_construct(queries, language_list, doc_list, device) if len(self.result_dict) == 0 else self._batch_construct_use_cache(queries, language_list, doc_list, device, world_size=world_size)
        
    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))
    
    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))

    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False,indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    def _construct(self, query, language, doc, gpu="gpu0"):
        pass

    def _batch_construct(self, queries, language_list, doc_list, gpu="gpu0"):
        return [self._construct(queries[index], language_list[index], doc_list[index], gpu) for index in tqdm(range(len(queries)))]

    @staticmethod
    def key(query, doc):
        key = f"query: {query}"
        if type(doc) is list:
            for i, d in enumerate(doc):
                key += f"\n\ndoc{i}: {d}"
        elif type(doc) is str:
            key += f"\n\ndoc: {doc}"
        return key

    def _batch_construct_use_cache(self, queries, language_list, doc_list, device="gpu0", world_size=1):
        cache = set(self.result_dict.keys())
        no_cache_queries = []
        no_cache_language_list = []
        no_cache_doc_list = []
        for query, language, doc in zip(queries, language_list, doc_list):
            if self.key(query, doc) not in cache:
                no_cache_queries.append(query)
                no_cache_language_list.append(language)
                no_cache_doc_list.append(doc)
        if len(no_cache_queries) > 0:
            self._batch_construct(no_cache_queries, no_cache_language_list, no_cache_doc_list, device)
        return [self.result_dict[self.key(query, doc)] for query, doc in zip(queries, doc_list)]