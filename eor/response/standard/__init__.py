from ..base import BaseGenerator
from ...utils import PromptTemplate, LLM
import copy


zh_qa_template = {
    1: "{query}",
    2: "你是一个高度智能的问答机器人, 如果你被问到了有事实依据的问题，你可以给我一个**简洁、短并且精确的答案**，如果你被问到了难以回答的问题，你可以回答不知道。\n\n假设以下事实是正确的：\n\n{knowledge}\n\n请回答以下问题：{query}\n\n",
}

en_qa_template = {
    1: "Assuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following question with one or few words:\n{query}",
    2: "Please directly answer the following question with one or few words:\n{query}",
    3: "Assuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following question within 15 words:\n{query}",
    4: "Please directly answer the following question within 15 words:\n{query}"
}

zh_system_message = "你是一个非常智能的问答机器人。有过你被问了有有事实依据的问题，你可以给出你的答案，如果你被问了难以回答的问题，你可以回答你不知道。"
en_system_message = {0: None,
                    1: "You are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me the answer. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'",
                    2: "You are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer with one or few words **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'."}

class StandardGenerator(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        
        if config["system_id"] is not None:
            self.use_system_message = True
        else:
            self.use_system_message = False
            config["system_id"] = 0

        self.config = config
        self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
        


    def build_prompt_template(self, config):
        

        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_qa_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="qa",
                                            template_id=config["zh_template_id"], use_system_message=self.use_system_message)
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_qa_template[config["en_template_id"]],
                                            system_message=en_system_message[config["system_id"]], task_name="qa",
                                            template_id=config["en_template_id"],
                                            use_system_message=self.use_system_message) 
        
                                            
            
        return zh_prompt_template, en_prompt_template

    def _response(self, query, language, knowledge, device="gpu0"):

        if self.query_only:
            knowledge = None
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "knowledge": knowledge})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "knowledge": knowledge})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["device_name"] = device
        
        response = LLM.lm_generate(**kwargs)[0]
        
        r = response
        while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
            if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                r = r[1:]
            if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                r = r[3:]
        response = r
        
        self.result_dict[self.key(query, knowledge)] = response
        
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"knowledge": knowledge, "fill": fill_info, "response": response})

        return response

    def _batch_response(self, query_list, language_list, knowledge_list, device="gpu0"):
        prompts = []
        infos = []
        if self.query_only:
            knowledge_list = [None] * len(query_list)
        for query, language, knowledge in zip(query_list, language_list, knowledge_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "knowledge": knowledge})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "knowledge": knowledge})

            prompts.append(prompt)
            infos.append(fill_info)
                


        kwargs = copy.deepcopy(self.config)
        kwargs["prompts"] = prompts
        
        if "zh" in language_list and "en" not in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        elif "zh" not in language_list and "en" in language_list:
            kwargs["model_name"] = kwargs["en_model_name"]
        else:
            raise KeyError("Now not support mix 'en' and 'zh'.")
        
        kwargs["device_name"] = device
        kwargs["verbose"] = self.verbose
        
        response_list = LLM.lm_generate(**kwargs)
        if isinstance(response_list[0], list) and len(response_list[0]) == 1:
            response_list = [r[0] for r in response_list]
    
        for index, res in enumerate(response_list):
            if isinstance(res, str):
                res = self.clean_response(res)
            elif isinstance(res, list):
                res = [self.clean_response(r) for r in res]
            else:
                raise ValueError("Something is wrong!")

            response_list[index] = res
    

        for query, knowledge, fill_info, response in zip(query_list, knowledge_list, infos, response_list):
            self.result_dict[self.key(query, knowledge)] = response
            self.renewed_result_dict[self.key(query, knowledge)] = response
                
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"knowledge": knowledge, "fill": fill_info, "response": response})

        return response_list
    
    def set_query_only_mode(self):
        if not self.config["dynamic_retrieval"]:
            raise AttributeError("this generator module do not support query only mode, please set 'dynamic_retrieval' to True")

        self.en_prompt_template = PromptTemplate(language="en", model_name=self.config["en_model_name"],
                                            template=en_qa_template[self.config["en_query_only_template_id"]],
                                            system_message=en_system_message[self.config["system_id"]], task_name="qa",
                                            template_id=self.config["en_query_only_template_id"], use_system_message=self.use_system_message)
        self.zh_prompt_template = PromptTemplate(language="zh", model_name=self.config["zh_model_name"],
                                            template=zh_qa_template[self.config["zh_query_only_template_id"]],
                                            system_message=zh_system_message, task_name="qa",
                                            template_id=self.config["zh_query_only_template_id"], use_system_message=self.use_system_message)
        self.query_only = True
    
    def reset_query_only_mode(self):
        self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(self.config)
        self.query_only = False
    
    def clean_response(self, r: str):
        r = r.strip()
        while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
            if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                r = r[1:]
            if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                r = r[3:]

        # clean response following llama2-chat habits
        if r.startswith("Sure!") or r.startswith("Sure,"):
            if len(r.split("\n\n", 1))>1:
                r = r.split("\n\n", 1)[-1]
            elif r.startswith("Sure! Based on"):
                r = r.split(",", 1)[-1]
        
        if r == "":
            r = "[EMPTY_ANSWER]"
        return r
            
