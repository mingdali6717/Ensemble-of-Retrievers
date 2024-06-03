import json
import torch

from loguru import logger


class BaseVoter:
    def __init__(self, config, result_dict=None):
        self.result_dict = {}
        self.renewed_result_dict = {}
        self.info_dict = {}
        self.verbose = config["verbose"]
        
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]
        
        if config["load_result_path"] is not None and result_dict is None:
            if self.verbose:
                logger.info(f"[Load voting from {config['load_result_path']}]")
            self.load_result(config["load_result_path"])
        elif result_dict is not None:
            self.result_dict = result_dict
        
        # if config["load_info_path"] is not None:
            # logger.warning(f"[Load voting info from {config['load_info_path']}]")
            # self.load_info(config["load_info_path"])

    def voting(self, query, language, responses, gpu="gpu0"):
        return self.result_dict[self.key(query, responses)] if self.key(query, responses) in self.result_dict.keys() else self._voting(query, language, responses, gpu)

    def batch_voting(self, queries, language_list, responses_list, gpu="gpu0"):
        return self._batch_voting(queries, language_list, responses_list, gpu) if len(self.result_dict) == 0 else self._batch_voting_use_cache(queries, language_list, responses_list, gpu)
    
    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))
        
    
    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))

    def save_if_set(self, new_only=False):
        if new_only:
            to_save = self.renewed_result_dict
        else:
            to_save = self.result_dict
        if self.save_result:
            json.dump(to_save, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    def _voting(self, query, language, responses, gpu="gpu0"):
        return self._batch_voting([query], [language], [responses], gpu)

    def _batch_voting(self, queries, language_list, responses_list, gpu="gpu0"):
        pass

    @staticmethod
    def key(query, hyp, ref, method):
        key = f"method:{method}\nquery: {query}\nhyp: {hyp}\nref: {ref}"
        
        return key
    