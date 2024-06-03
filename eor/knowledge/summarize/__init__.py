from ..base import BaseConstructor
from ...utils import LLM, PromptTemplate, OPENAI_MODEL_LIST
from loguru import logger
import torch.distributed as dist
import torch


zh_summarization_template = {
    1: "请如实总结以下文档，总结中应该包含用于回答问题的最主要的信息且字数在200字以内：\n\n问题: {query}\n\n文档：{doc}\n\n总结: ",
    
}

# Template 4 is the best after test.
en_summarization_template = {
    1: "Please truthfully summarize the document below, the summary should contain the most important information relevant to answer the query and be within 200 words:\n\nquery: {query}\n\ndocument: {doc}\n\nsummary: "}

zh_knowledge_length = {
    1: 200,
}

en_knowledge_length = {
    1: 200,
}

zh_system_message = "你是一个非常有帮助的文档总结助手，你可以从文档中提取出最有价值的信息。"
en_system_message = "You are an incredibly helpful document summarization assistant capable of extracting the most valuable information from documents."


class SummarizeConstructor(BaseConstructor):
    def __init__(self, config):
        super().__init__(config)

        
        self.config = config
        self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
        self.zh_knowledge_length = zh_knowledge_length[config["zh_template_id"]]
        self.en_knowledge_length = en_knowledge_length[config["en_template_id"]]

    def _construct(self, query, language, doc, device="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_knowledge_length if language == "zh" else self.en_knowledge_length
        
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["device_name"] = device
        
        knowledge = LLM.lm_generate(**kwargs)[0]
        if type(knowledge) is list:
            knowledge = knowledge[0]

        r = knowledge
        while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
            if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                r = r[1:]
            if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                r = r[3:]
        knowledge = r

        self.result_dict[self.key(query, doc)] = knowledge
        
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"doc": doc, "fill": fill_info, "knowledge": knowledge, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"]})

        return knowledge

    def _batch_construct(self, query_list, language_list, doc_list, device="gpu0"):
        prompts = []
        infos = []
        for query, language, doc in zip(query_list, language_list, doc_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})

            prompts.append(prompt)
            infos.append(fill_info)

        kwargs = self.config
        kwargs["prompts"] = prompts
        
        if "zh" in language_list and "en" in language_list:
            assert False, "it's recommended to divide chinese and english queries into two individual parts."
        elif "zh" in language_list and "en" not in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_knowledge_length
        elif "zh" not in language_list and "en" in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.en_knowledge_length       
        
        if "zh" in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["device_name"] = device
        kwargs["verbose"] = self.verbose
        
        knowledge_list = LLM.lm_generate(**kwargs)  

        if isinstance(knowledge_list[0], list) and len(knowledge_list[0]) == 1:
            knowledge_list = [r[0] for r in knowledge_list]          
        
        for index, r in enumerate(knowledge_list):
            while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
                if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                    r = r[1:]
                if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                    r = r[3:]
            knowledge_list[index] = r
        
        for query, language, doc, fill_info, knowledge in zip(query_list, language_list, doc_list, infos, knowledge_list):
            self.result_dict[self.key(query, doc)] = knowledge
            
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"doc": doc, "fill": fill_info, "knowledge": knowledge, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"]})

        return knowledge_list

    @staticmethod
    def build_prompt_template(config):
        use_system_message = False
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_summarization_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="summarize",
                                            template_id=config["zh_template_id"],use_system_message=use_system_message)
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_summarization_template[config["en_template_id"]],
                                            system_message=en_system_message, task_name="summarize",
                                            template_id=config["en_template_id"], use_system_message=use_system_message)
        return zh_prompt_template, en_prompt_template
    
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

        if len(no_cache_queries) > 0 and self.verbose:
            logger.info(f"{device}: {len(no_cache_queries)} non-cached queries are found, start to summarize from scratch.")
        
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
            self._batch_construct(no_cache_queries, no_cache_language_list, no_cache_doc_list, device)
        return [self.result_dict[self.key(query, doc)] for query, doc in zip(queries, doc_list)]
