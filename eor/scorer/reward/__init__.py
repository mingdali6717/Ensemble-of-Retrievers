from eor.scorer.base import BaseScorer
from eor.utils import PromptTemplate, LLM

zh_rm_template = {
    1: "请对助手回复的有用性、事实性进行打分。\n\n人类：{query}\n\n助手：{answer}",
}

# 因为英文用的是Llama2-chat，而中文是xverse-13B，是base版
en_rm_template = {
    1: "[INST] <<SYS>>\nScore the faithfulness and helpfulness of assistant's response.\n<</SYS>>\n\n{query} [/INST] [INST] {answer} [/INST] [INST] Score the faithfulness and helpfulness range from -1 to 1 [/INST]",
}


class RewardScorer(BaseScorer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
    
    @staticmethod
    def build_prompt_template(config):
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_rm_template[config["zh_template_id"]],
                                            system_message=None, task_name="rm",
                                            template_id=config["zh_template_id"])
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_rm_template[config["en_template_id"]],
                                            system_message=None, task_name="rm",
                                            template_id=config["en_template_id"])
        return zh_prompt_template, en_prompt_template
    
    def _score(self, query, language, response, gpu="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "answer": response})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "answer": response})
        prompts = [prompt]
        
        kwargs = self.config
        model_name = kwargs["zh_model_name"] if  language == "zh" else kwargs["en_model_name"]

        score = LLM.lm_reward(model_name, prompts, gpu, kwargs["tokenize_kwargs"], kwargs["reward_kwargs"])[0]
        
        self.result_dict[self.key(query, response)] = score
        if query not in self.info_dict:
            self.info_dict[query] = []
        self.info_dict[query].append({"response": response, "fill": fill_info, "score": score})
        
        return score

    def _batch_score(self, queries, language_list, response_list, gpu="gpu0"):
        score_list = []
        
        prompts = []
        infos = []
        for query, language, response in zip(queries, language_list, response_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "answer": response})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "answer": response})

            prompts.append(prompt)
            infos.append(fill_info)
        
        kwargs = self.config
        if "zh" in language_list and "en" not in language_list:
            model_name = kwargs["zh_model_name"]
        elif "zh" not in language_list and "en" in language_list:
            model_name = kwargs["en_model_name"]
        else:
            assert False, "Now not support mix 'en' and 'zh'."
        
        score_list = LLM.lm_reward(model_name, prompts, gpu, kwargs["tokenize_kwargs"], kwargs["reward_kwargs"])
                
        for query, response, fill_info, score in zip(queries, response_list, infos, score_list):
            self.result_dict[self.key(query, response)] = score
            if query not in self.info_dict:
                self.info_dict[query] = []
            self.info_dict[query].append({"response": response, "fill": fill_info, "score": score})

        return score_list