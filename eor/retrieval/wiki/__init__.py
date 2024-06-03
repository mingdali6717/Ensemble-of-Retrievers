import json
import numpy as np
import faiss
import torch

from zhconv import convert

from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from loguru import logger

from ..base import BaseRetriever
from ...utils import LLM


class WikiRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

        
        # Lazy initialization
        self.zh_searcher = None
        self.en_searcher = None
        self.k=20
        self.config = config

    def initialize(self):
        documents = json.load(open("./data/index/documents.json", "r"))
        documents = [documents[i]["contents"] for i in range(len(documents))]
        index = faiss.IndexFlatIP(1024)
        index.add(np.load("./data/index/data.npy"))
        instruction = "为这个句子生成表示以用于检索相关文章："
        self.zh_searcher = {"documents": documents, "index": index, "instruction": instruction}
        self.en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))

    def _retrieve(self, query, language, device="gpu0"):
        _docs, _info = [], []
        if language == "en":
            if self.en_searcher is None:
                self.en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))
                if self.log_detail:

                    logger.info("FaissSearcher loaded, start to retrieve from wiki corpus!!!")
            
            
            hits = self.en_searcher.search(query, k=self.k)
            for i in range(0, self.k):
                doc = json.loads(self.en_searcher.doc(hits[i].docid).raw())["contents"]
                _docs.append(doc)
                _info.append({"score": hits[i].score.item(), "doc": doc})
        else:
            if self.zh_searcher is None:
                documents = json.load(open("./data/index/documents.json", "r"))
                documents = [documents[i]["contents"] for i in range(len(documents))]
                index = faiss.IndexFlatIP(1024)
                index.add(np.load("./data/index/data.npy"))
                instruction = "为这个句子生成表示以用于检索相关文章："
                self.zh_searcher = {"documents": documents, "index": index, "instruction": instruction}
            
            kwargs = self.config
            
            _query = self.zh_searcher["instruction"] + query
            
            query_emb = LLM.lm_encode("zh_encoder", [_query], device, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
            
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1).detach().cpu().numpy()

            _, I = self.zh_searcher["index"].search(query_emb, self.k)

            docs = [convert(self.zh_searcher["documents"][i], "zh-cn") for i in I[0]]

            for i in range(0, self.k):
                _docs.append(docs[i])
                _info.append({"doc": docs[i]})
        
        self.result_dict[query] = _docs
        self.info_dict[query] = _info

        return _docs

    # def _batch_retrieve(self, queries, language_list, device="gpu0"):
        # if isinstance(queries, str):
            # queries = [queries]
# 
        # if language_list[0] == "en":
            # if self.en_searcher is None:
                # self.en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))
                # if self.log_detail:
                    # logger.info("FaissSearcher loaded, start to retrieve from wiki corpus!!!")
# 
# 
            # hits = self.en_searcher.batch_search(queries, queries, k=self.k, return_vector=False, threads=20)
            # _docs = []
            # for query in queries:
                # docs = []
                # info = []
                # hit = hits[query]
                # for i in range(len(hit)):
                    # doc = json.loads(self.en_searcher.doc(hit[i].docid).raw())["contents"]
                    # docs.append(doc)
                    # info.append({"score": hit[i].score.item(), "doc": doc})
                # self.result_dict[query] = docs
                # self.info_dict[query] = info
# 
                # _docs.append(docs)
            # return _docs
# 
        # else:
            # if self.zh_searcher is None:
                # documents = json.load(open("./data/index/documents.json", "r"))
                # documents = [documents[i]["contents"] for i in range(len(documents))]
                # index = faiss.IndexFlatIP(1024)
                # index.add(np.load("./data/index/data.npy"))
                # instruction = "为这个句子生成表示以用于检索相关文章："
                # self.zh_searcher = {"documents": documents, "index": index, "instruction": instruction}
# 
            # kwargs = self.config
# 
            # _query = [self.zh_searcher["instruction"] + q for q in queries]
# 
            # query_emb = LLM.lm_encode("zh_encoder", _query, device, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
# 
            # query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1).detach().cpu().numpy()
            # _, I = self.zh_searcher["index"].search(query_emb, queries, 10)
            # _docs = []
            # for query_id, docs_id in enumerate(I):
                # info = []
                # docs = [convert(self.zh_searcher["documents"][i], "zh-cn") for i in docs_id]
                # for i in range(0, len(docs)):
                # 
                    # info.append({"doc": docs[i]})
# 
                # self.result_dict[queries[query_id]] = docs
                # self.info_dict[queries[query_id]] = info
# 
                # _docs.append
        # return _docs