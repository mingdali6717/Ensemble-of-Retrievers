from itertools import chain
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import json
import os
import re
import time
import datetime
from ..retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from ..knowledge import ContrieverConstructor, SummarizeConstructor
from transformers import LlamaTokenizer
import numpy as np

from ..utils import truncate_en_doc, truncate_zh_doc

METHOD_MODULE_MAPPING = {
    "google_best": {
        "retriever":["Web"],
        "summarizer": ["Summarizer"]
    },
    "wiki": {
        "retriever":["Wiki"],
        "summarizer": ["Summarizer"]
    },
    "gendoc":{
        "retriever":["Gendoc"],
        "summarizer": ["Summarizer"]
    },
    "google_contrieve":{
        "retriever":["Web"],
        "contriever": ["Contriever"],
        "summarizer": ["Summarizer"]
    },
    "google_merge": {
        "retriever":["Web"],
        "summarizer": ["Summarizer"]
    },
    "google_merge+wiki": {
        "retriever":["Web", "Wiki"],
        "summarizer": ["Summarizer"]
    },
    "google_contrieve+wiki": {
        "retriever":["Web", "Wiki"],
        "contriever": ["Contriever"],
        "summarizer": ["Summarizer"]
    },
    "contrieve_all_sources":{
        "retriever":["Web", "Wiki", "Gendoc"],
        "contriever": ["Contriever"],
        "summarizer": ["Summarizer"]
    },
}

class ControllerConstructor:

    def __init__(self, config, device, world_size = 1, rank=None):
        self.verbose = config["log_detail"]
        self.active_methods = config.active_methods
        self.modules = config.active_modules
        self.rank = rank
        self.config=config
        self.device = device
        self.ddp = config["ddp"]
        self.world_size = world_size

        self.required_retrieval_modules = sorted(list(config.retrieval_modules), reverse=True)
        self.required_contriever_modules = sorted(list(config.contriever_modules))
        self.required_summarizer_modules = sorted(list(config.summarizer_modules))

        self.max_knowledge_length = config["max_knowledge_length"]

        self.result_dict = dict()
        self.result_dict["active_methods"] = self.active_methods
        self.result_dict["processed_knowledge"] = dict()
        
        if config["output_dir"] is not None:
            self.save_result = True
            if "$rank$" in config["output_dir"] and rank is not None:
                self.save_result_path = os.path.join(config["output_dir"].replace("$rank$", str(rank)), "controller_result.json")
            else: 
                self.save_result_path = os.path.join(config["output_dir"], "controller_result.json")

        self.from_cache = config.load_controller_from_cache
        if self.from_cache:
            self.load_result_path = os.path.join(config["cache_dir"], "controller_result.json")


        log_name = config.get("log_name", config["time"]) + ".log"
        logger.add(os.path.join("log", log_name), level='INFO')
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')

        
    def batch_process(self, queries, language_list):
        if self.active_methods == dict():
            if self.rank == 0 or not self.ddp:
                logger.info("Controller do not retrieve and process any knowledge!")
            return None
        
        if self.from_cache:

            result_dict = json.load(open(self.load_result_path, "r", encoding="UTF-8"))

            
            if self.rank == 0 or not self.ddp:
                logger.info(f"load cached data from '{self.load_result_path}'")
            self.result_dict = result_dict
            cached = set(result_dict["processed_knowledge"].keys())
            no_cache_queries = [query for query in queries if query not in cached]
            if self.verbose:
                logger.info(f"{self.device}:{len(queries) - len(no_cache_queries)} cached queries found, {len(no_cache_queries)} more queries to be processed.")
            no_cache_language_list = [language for query, language in zip(queries, language_list) if query not in cached]
            if len(no_cache_queries) > 0:
                _ = self._batch_process(no_cache_queries, no_cache_language_list)
            processed_knowledge = defaultdict(list)
            for q in queries:
                for m_name, m_doc in self.result_dict["processed_knowledge"][q].items():
                    processed_knowledge[m_name].append(m_doc)
            return processed_knowledge
            
        else:
            if self.rank == 0 or not self.ddp:
                if self.verbose:
                    logger.info(f"No cached file found for controller. Start to retrieve and process knowledge from scratch.")
            return self._batch_process(queries, language_list)
                
            
    
    def _batch_process(self, queries, language_list):
        
        query_num = len(queries)
        
        #start to retrieve
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to retrieval, modules used to retrieve are {'|'.join(self.required_retrieval_modules)}************")
        
        retrieval_doc_list = self._batch_retrieve(queries, language_list)

        processed_knowledge = dict()

        retriever_processed = dict()
        retrieval_index = {name: self.build_retrieval_method_index(method) for name, method in self.active_methods.items()}
        contriever_index = dict()
        postprocess_index = dict()
        summarizer_index = dict()
        docs_to_contrieve = defaultdict(set) 
        docs_to_summarize = defaultdict(list)

        for name, method in self.active_methods.items():
            
            if retrieval_index[name] not in retriever_processed.keys():
                retriever_processed[retrieval_index[name]] = self._process_retrieved_docs(method, language_list, retrieval_doc_list)
            
            
            
            processed_knowledge[name] = retriever_processed[retrieval_index[name]]
            
            if method["contriever"] is not None:
                contriever_index[name] = retrieval_index[name]+"-"+method["contriever"][0]
                docs_to_contrieve[method["contriever"][0]].add(retrieval_index[name])
            
            if method["method_name"] == "google_contrieve+wiki":
                postprocess_index[name] = {
                    "method_name": "google_contrieve+wiki",
                    "contriever": contriever_index[name],
                    "retriever": self.get_module_names(method, "Wiki")
                }
            
            if method["summarizer"] is not None:
                docs_to_summarize[method["summarizer"][0]].append(name)


        #json.dump(retriever_processed, open(os.path.join(self.config["output_dir"], "retriever_processed.json"), 'w'), indent=4)
        docs_to_contrieve = {k: list(v) for k,v in docs_to_contrieve.items()}

        # if self.rank == 0 or not self.ddp:
            # logger.info(f"retrieval indexes:\n {json.dumps(retrieval_index,indent=4)}")
            # logger.info(f"contriever indexes:\n {json.dumps(contriever_index,indent=4)}")
            # logger.info(f"contriever_retriever_index_mapping:\n {json.dumps(docs_to_contrieve,indent=4)}")
# 
        #start to contrieve
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to contrieve, modules used to contrieve are '{'|'.join(list(self.required_contriever_modules))}'************")
        contriever_processed = dict()
        begin_time = time.time()
        for contriever_name, doc_names in docs_to_contrieve.items():
            Contriever = ContrieverConstructor(self.modules[contriever_name])
            doc_type = len(doc_names)
            docs = list(chain(*[retriever_processed[n] for n in doc_names]))
            qs = queries * doc_type
            ls = language_list * doc_type
            knowledges = Contriever.batch_construct(qs, ls, docs, device=self.device, top_k=10, world_size=self.world_size)
            Contriever.save_if_set()
            for idx, r_name in enumerate(doc_names):
                c_idx = r_name + "-" + contriever_name
                assert c_idx not in contriever_processed.keys(), "something is wrong"
                contriever_processed[c_idx] = knowledges[query_num*idx : query_num*(idx+1)]
            del Contriever
        
        for name, c_index in contriever_index.items():
            processed_knowledge[name] = contriever_processed[c_index]
        
        if self.verbose:
            logger.info(f"********************{self.device}: contriever finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}")

        #start to postprocess retrieval and contriever docs
        for name, p_index in postprocess_index.items():
            contriever_docs = [d.split("\n") for d in contriever_processed[p_index["contriever"]]]
            retrieval_docs = retrieval_doc_list[p_index["retriever"]]
            processed_knowledge[name] = self._postprocess_docs(p_index["method_name"],contriever_docs, retrieval_docs, language_list)
        
        #truncate all knowledge based on max_knowledge_length
        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to truncate long knowledge to maximum knowledge length {self.max_knowledge_length}************")

        if self.config["tokenizer"] == "llama":
            tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")
        elif self.config["tokenizer"] is None or self.config["tokenizer"] == "nltk":
            tokenizer = self.config["tokenizer"]
        else:
            raise KeyError("tokenizer should be one of None, llama, nltk.")
        
        truncated_knowledge = dict()
        for n, k in processed_knowledge.items():
            new_k = []
            for d,lang in zip(k, language_list):
                new_k.append(truncate_doc[lang](" ".join(re.sub(r'[\n]+', '\n', d).split()), max_doc_len = self.max_knowledge_length, min_doc_len = 0, tokenizer=tokenizer))
            truncated_knowledge[n] = new_k
        


        
        # start to summarize
        if self.verbose:
            logger.info(f"************{self.device}: Start to summarize, modules used to summarize are '{'|'.join(list(self.required_summarizer_modules))}'************")
        
        begin_time = time.time()
        for summarizer_name, methods_to_sum in docs_to_summarize.items():
            Summarizer = SummarizeConstructor(self.modules[summarizer_name])
            doc_type = len(methods_to_sum)
            docs = list(chain(*[truncated_knowledge[n] for n in methods_to_sum]))
            qs = queries * doc_type
            ls = language_list * doc_type
            assert len(docs) == doc_type * query_num, "something wrong"
            
            
            knowledges = Summarizer.batch_construct(qs, ls, docs, device=self.device, world_size=self.world_size)
            Summarizer.save_if_set()
            for idx, m_name in enumerate(methods_to_sum):
    
                truncated_knowledge[m_name] = knowledges[query_num*idx : query_num*(idx+1)]

            del Summarizer
        if self.verbose:
            logger.info(f"********************{self.device}: summarization finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}")

        
        
        for idx, query in enumerate(queries):
            collected_k = dict()
            for m_name, ds in truncated_knowledge.items():
                collected_k[m_name] = ds[idx]
            self.result_dict["processed_knowledge"][query] = collected_k

        return truncated_knowledge            

    
    def _batch_retrieve(self, queries, language_list):

        retrieval_doc_list = dict()
        begin_time = time.time()
        for r_name in self.required_retrieval_modules:
            if self.modules[r_name]["type"] == "Web":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Web Retriever: {r_name}************")
                retriever = WebRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries,language_list)
                doc_list = [[d for d in docs if not (np.mean([len(s.split()) for s in re.split(r"([.!?;,])", d)]) < 1.5 and len(d.split()) > 30)] for docs in doc_list]
                if self.verbose:
                    logger.info(f"************{self.device}: Web Retriever Finished************")
                
            elif self.modules[r_name]["type"] == "Wiki":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Wiki Retriever: {r_name}************")
                retriever = WikiRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list)
                if self.verbose:
                    logger.info(f"************{self.device}: Wiki Retriever Finished************")

            elif self.modules[r_name]["type"] == "Gendoc":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Gendoc Retriever: {r_name}************")
                retriever = GenerationRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list, self.device, world_size=self.world_size)
                if isinstance(doc_list[0], list):
                    doc_list = ["\n".join(d) for d in doc_list]
                if self.verbose:
                    logger.info(f"************{self.device}: Gendoc Retriever Finished************")
            else:
                raise TypeError(f"retrieval modules should be one of Web, Wiki or Gendoc, but '{r_name}' is given")

            retriever.save_if_set()
            retrieval_doc_list[r_name] = doc_list

            del retriever
        if self.verbose:
            logger.info(f"********************{self.device}: Retrieval finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}********************")

        return retrieval_doc_list
    
    def _process_retrieved_docs(self, config, language_list, retrieved_docs):

        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }

        if config["method_name"] == "google_best":
            web_module_name = self.get_module_names(config, "Web")

            return [truncate_doc[lang](doc_list[0], 1000, 0) if len(doc_list) > 0 else "" for lang, doc_list in zip(language_list, retrieved_docs[web_module_name])]
        elif config["method_name"] == "wiki":
            wiki_module_name = self.get_module_names(config, "Wiki")

            return ["\n".join(doc_list[:10]) for doc_list in retrieved_docs[wiki_module_name]]
        elif config["method_name"] == "gendoc":
            gen_module_name = self.get_module_names(config, "Gendoc")
            
            return retrieved_docs[gen_module_name]
        elif config["method_name"] == "google_contrieve" or config["method_name"] == "google_contrieve+wiki":
            web_module_name = self.get_module_names(config, "Web")

            return ["\n".join(doc_list) for doc_list in retrieved_docs[web_module_name]]
        elif config["method_name"] == "google_merge":
            web_module_name = self.get_module_names(config, "Web")
            docs = retrieved_docs[web_module_name]
            new_docs = []
            for doc, lang in zip(docs, language_list):
                new_docs.append("\n".join([truncate_doc[lang](d, 250) for d in doc[:4]]))
            return new_docs
        elif config["method_name"] == "google_merge+wiki":
            web_module_name = self.get_module_names(config, "Web")
            wiki_module_name = self.get_module_names(config, "Wiki")
        
            web_docs = retrieved_docs[web_module_name]
            wiki_docs = retrieved_docs[wiki_module_name]
            new_docs = []
            for web, wiki, lang in zip(web_docs, wiki_docs, language_list):
                new_docs.append("\n".join([truncate_doc[lang](d, 250) for d in web[:2]]) + "\n" + "\n".join(wiki[:5]))
            return new_docs
        elif config["method_name"] == "contrieve_all_sources":
            web_module_name = self.get_module_names(config, "Web")
            wiki_module_name = self.get_module_names(config, "Wiki")
            gen_module_name = self.get_module_names(config, "Gendoc")
            web_docs = retrieved_docs[web_module_name]
            wiki_docs = retrieved_docs[wiki_module_name]
            gen_docs = retrieved_docs[gen_module_name]
            new_docs = []
            for web, wiki, gen in zip(web_docs, wiki_docs, gen_docs):
                new_docs.append("\n".join(web) + "\n" + "\n".join(wiki) + "\n" + gen)
            return new_docs
        else:
            raise TypeError(f"method {config['method_name']} is not supported.")
    
    def _postprocess_docs(self, method_name, contrieved_docs, retrieved_docs, language_list):
        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }
        if method_name == "google_contrieve+wiki":
            post_docs = []
            for c_d, r_d, lang in zip(contrieved_docs, retrieved_docs, language_list):
                post_docs.append(truncate_doc[lang]("\n".join(c_d[:5]), 500) + "\n" + truncate_doc[lang]("\n".join(r_d[:5]), 500))
            
            return post_docs
        else:
            raise TypeError(f"{method_name} is not supported for postprocessing")


    
    def build_retrieval_method_index(self, config):
        """
        return a index to identify processed retrieval docs.
        index in a format {method_name}-"-".join({used_retriever_module_names})
        """
        if config["method_name"] == "google_contrieve+wiki":
            return "google_contrieve" + "-" + self.get_module_names(config, "Web")

        else:
            return config["method_name"] + "-" + "-".join(sorted( config["retriever"]))
        
    
    def get_module_names(self, method_config, module_type):
        """
        given a method config and the required module name, return the corresponding module name and check uniqueness.
        for example:
        config:
        {
        "method_name": "gendoc",
        "retriever": ["Gendoc_1"]
        }

        when given type Gendoc, will return "Gendoc_1"
        """
        if module_type in ["Web", "Wiki", "Gendoc"]:
            module_name = [n for n in method_config["retriever"] if self.modules[n]["type"]==module_type]
        elif module_type == "Contriever":
            module_name = [n for n in method_config["contriever"] if self.modules[n]["type"]==module_type]
        elif module_type == "Summarizer":
            module_name = [n for n in method_config["summarizer"] if self.modules[n]["type"]==module_type]
        else:
            raise TypeError(f"given module type '{module_type} is not supported")
        
        return module_name[0]

    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
    
    def load_from_cache(self, queries, result_dict, knowledge_names):
        processed_knowledge = defaultdict(list)
        for q in queries:
            for m_name in knowledge_names:
                if m_name == "query_only":
                    continue
                processed_knowledge[m_name].append(result_dict["processed_knowledge"][q][m_name])
        return processed_knowledge





    
    
    