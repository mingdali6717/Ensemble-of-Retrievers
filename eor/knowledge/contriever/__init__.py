import re
import time

from ..base import BaseConstructor
from ...utils import truncate_en_doc, truncate_zh_doc, split_with_fix_length_zh, split_with_fix_length_en, LLM, OPENAI_MODEL_LIST
from loguru import logger
import torch.distributed as dist
import torch

class ContrieverConstructor(BaseConstructor):
    def __init__(self, config):
        super().__init__(config)

        
        self.config = config
        self.min_knowledge_len = config["min_knowledge_len"]

    def _construct(self, query, language, doc, gpu="gpu0"):
        if type(doc) is list:
            assert False, "If you have many docs for a query, use batch_construct."

        paragraphs = []
        for item in doc.split("\n"):
            item = item.strip()
            if not item:
                continue
            paragraphs.append(item)

        temp = []
        for para in paragraphs:
            para = para.strip()
            para = re.sub(r"\[\d+\]", "", para)
            if language == "en":
                para = split_with_fix_length_en(para, fix_length=100)
            else:
                para = split_with_fix_length_zh(para, fix_length=100)
            if para is None:
                continue
            temp.extend(para)
        paragraphs = temp
        
        kwargs = self.config
        
        if language == "en":
            query_model_name = kwargs["en_query_model_name"]
            paragraph_model_name = kwargs["en_paragraph_model_name"]
            instruction = ""
        elif language == "zh":
            query_model_name = kwargs["zh_query_model_name"]
            paragraph_model_name = kwargs["zh_paragraph_model_name"]
            instruction = "为这个句子生成表示以用于检索相关文章："
            kwargs["encode_kwargs"]["pooling_method"] = "cls"     
           
        query_embedding = LLM.lm_encode(query_model_name, [instruction + query], gpu, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])[0]
        paragraph_embeddings = LLM.lm_encode(paragraph_model_name, paragraphs, gpu ,kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])

        scores = query_embedding @ paragraph_embeddings.t()

        sorted_paragraphs = sorted(paragraphs, key=lambda p: scores[paragraphs.index(p)], reverse=True)

        knowledge = "\n".join(sorted_paragraphs[:3])

        if len(knowledge.split(" ")) * 2 < self.min_knowledge_len and len(sorted_paragraphs) >= 4:
            knowledge += "\n" + sorted_paragraphs[3]

        # knowledge = truncate_en_doc(knowledge, self.max_doc_len, self.min_doc_len)

        self.result_dict[self.key(query, doc)] = knowledge
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"doc": doc, "result": knowledge})

        return knowledge
    
    def batch_construct(self, queries, language_list, doc_list, device="gpu0", world_size=1, top_k=10):
        return self._batch_construct(queries, language_list, doc_list, device, top_k=top_k) if len(self.result_dict) == 0 else self._batch_construct_use_cache(queries, language_list, doc_list, device, top_k=top_k, world_size=world_size)

    def _batch_construct(self, queries, language_list, doc_list, device="gpu0", top_k = 10):
        """
        queries: List[str]
        doc_List: List[List[str]], each paragraph separated by '\n' in each doc
        """
        paragraphs_list = []
        paragraphs_index = [0]
        
        for language, doc in zip(language_list, doc_list):
            paragraphs = []
            for item in doc.split("\n"):
                item = item.strip()
                if not item:
                    continue
                paragraphs.append(item)

            temp = []
            for para in paragraphs:
                para = para.strip()
                para = re.sub(r"\[\d+\]", "", para)
                if language == "en":
                    para = split_with_fix_length_en(para)
                else:
                    para = split_with_fix_length_zh(para)
                if para is None:
                    continue
                temp.extend(para)
            paragraphs = temp
            
            paragraphs_list.append(paragraphs)
            paragraphs_index.append(paragraphs_index[-1] + len(paragraphs))
        
        kwargs = self.config
        
        if language == "en":
            query_model_name = kwargs["en_query_model_name"]
            paragraph_model_name = kwargs["en_paragraph_model_name"]
            instruction = ""
        elif language == "zh":
            query_model_name = kwargs["zh_query_model_name"]
            paragraph_model_name = kwargs["zh_paragraph_model_name"]
            instruction = "为这个句子生成表示以用于检索相关文章："
            kwargs["encode_kwargs"]["pooling_method"] = "cls" 
        
        queries_embedding = LLM.lm_encode(query_model_name, [instruction + q for q in queries], device, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
        paragraphs_embeddings = LLM.lm_encode(paragraph_model_name, [p for ps in paragraphs_list for p in ps], device, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
        
        paragraphs_embedding_list = []
        for i in range(len(queries)):
            paragraphs_embedding_list.append(paragraphs_embeddings[paragraphs_index[i]: paragraphs_index[i+1]])

        knowledge_list = []
        for query_embedding, paragraphs_embedding, query, doc, paragraphs in zip(queries_embedding, paragraphs_embedding_list, queries, doc_list, paragraphs_list):     
            scores = query_embedding @ paragraphs_embedding.t()
            sorted_paragraphs = sorted(paragraphs, key=lambda p: scores[paragraphs.index(p)], reverse=True)

            knowledge = "\n".join(sorted_paragraphs[:top_k])

            if len(knowledge.split(" ")) * 2 < self.min_knowledge_len and len(sorted_paragraphs) >= top_k+1:
                knowledge += "\n" + sorted_paragraphs[top_k]

            # knowledge = truncate_en_doc(knowledge, self.max_knowledge_len, self.min_knowledge_len)

            knowledge_list.append(knowledge)
            
            self.result_dict[self.key(query, doc)] = knowledge
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"doc": doc, "result": knowledge})

        return knowledge_list
    
    def _batch_construct_use_cache(self, queries, language_list, doc_list, device="gpu0", top_k = 10, world_size=1):
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
            logger.info(f"{device}: {len(no_cache_queries)} non-cached queries are found, start to contrieve from scratch.")
        # initialize all model to circumvate dist.barrier() in DistributedDataParralism
        if language_list[0] == "zh":
            query_model_name = self.config["zh_query_model_name"]
            paragraph_model_name = self.config["zh_paragraph_model_name"]
        else:
            query_model_name = self.config["en_query_model_name"]
            paragraph_model_name = self.config["en_paragraph_model_name"]
        

        if world_size > 1:
            gathered_tensor = torch.zeros(world_size, dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
            if len(no_cache_queries) > 0:
                have_non_cached = torch.tensor([1], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
            else: 
                have_non_cached = torch.tensor([0], dtype=torch.int64, device = torch.device(f"cuda:{device.replace('gpu','')}"))
            dist.all_gather_into_tensor(gathered_tensor, have_non_cached)
            
            if gathered_tensor.sum().item() > 0:
                
                LLM.initial_lm(query_model_name, device)
                LLM.initial_lm(paragraph_model_name, device)


        if len(no_cache_queries) > 0:
            self._batch_construct(no_cache_queries, no_cache_language_list, no_cache_doc_list, device, top_k = top_k)
        return [self.result_dict[self.key(query, doc)] for query, doc in zip(queries, doc_list)]