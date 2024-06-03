from eor.retrieval.base import BaseRetriever
from eor.utils import LLM, PromptTemplate, OPENAI_MODEL_LIST
import torch
import torch.distributed as dist
from loguru import logger


zh_generation_template = {
    1: "请给出回答'{query}'所必须的信息。\n\n",
}

en_generation_template = {
    1: "Generate a background document to answer the given question.\n{query}",
}

zh_doc_length = {
    1: 300,
}

en_doc_length = {
    1: 300, 
}

zh_system_message = "你是一个非常有帮助的助手，你可以给出回答问题所需要的信息，从而帮助他人给出回答。"
en_system_message = "You are an incredibly helpful information generation assistant. You can give the information needed to answer the question, thereby helping others to give an answer."


class GenerationRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

    
        self.config = config
        self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
        self.zh_doc_length = zh_doc_length[config["zh_template_id"]]
        self.en_doc_length = en_doc_length[config["en_template_id"]]

    def _retrieve(self, query, language, device="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_doc_length if language == "zh" else self.en_doc_length
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["device_name"] = device
        
        doc = LLM.lm_generate(**kwargs)[0]

        self.result_dict[query] = doc
        self.info_dict[query] = {"fill_info": fill_info, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"], "gen doc": doc}

        return doc

    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        prompts = []
        infos = []
        for query, language in zip(queries, language_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query})

            prompts.append(prompt)
            infos.append(fill_info)

        kwargs = self.config
        kwargs["prompts"] = prompts

        if "zh" in language_list and "en" in language_list:
            # kwargs["max_tokens"] = min(self.zh_doc_length, self.en_doc_length)
            assert False, "it's recommended to divide chinese and english queries into two individual parts."
        elif "zh" in language_list and "en" not in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_doc_length
        elif "zh" not in language_list and "en" in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.en_doc_length

        if "zh" in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["device_name"] = device
        kwargs["verbose"] = self.log_detail
        doc_list = LLM.lm_generate(**kwargs)

        for query, language,fill_info, doc in zip(queries, language_list, infos, doc_list):
            if self.save_result:
                self.result_dict[query] = doc
            if self.save_info:
                self.info_dict[query] = {"fill_info": fill_info, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"], "gen doc": doc}

        return doc_list
    
    def _batch_retrieve_use_cache(self, queries, language_list, device="gpu0", world_size=1):
        cache = set(self.result_dict.keys())
        no_cache_queries = [query for query in queries if query not in cache]
        no_cache_language_list = [language for query, language in zip(queries, language_list) if query not in cache]

        if len(no_cache_queries) > 0 and self.log_detail:
            logger.info(f"{device}: {len(no_cache_queries)} non-cached queries are found, start to generate documents from scratch.")

        if language_list[0] == "zh":
            model_name = self.config["zh_model_name"]
        else:
            model_name = self.config["en_model_name"]
        
        if model_name not in OPENAI_MODEL_LIST:
            # initialize all model
        
            if world_size > 1:
                gathered_tensor = torch.zeros(world_size, dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
                if len(no_cache_queries) > 0:
                    have_non_cached = torch.tensor([1], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
                else: 
                    have_non_cached = torch.tensor([0], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))

                dist.all_gather_into_tensor(gathered_tensor, have_non_cached)

                if gathered_tensor.sum().item() > 0:
                    
                    LLM.initial_lm(model_name, device, self.log_detail)

        if len(no_cache_queries) > 0:
            self._batch_retrieve(no_cache_queries, no_cache_language_list, device)
        return [self.result_dict[query] for query in queries]

    @staticmethod
    def build_prompt_template(config):
        use_system_mesage = False
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_generation_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="gendoc",
                                            template_id=config["zh_template_id"],use_system_message=use_system_mesage)
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_generation_template[config["en_template_id"]],
                                            system_message=en_system_message, task_name="gendoc",
                                            template_id=config["en_template_id"], use_system_message=use_system_mesage)
        return zh_prompt_template, en_prompt_template
