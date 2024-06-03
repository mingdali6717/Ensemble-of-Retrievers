import re
import emoji
from loguru import logger
from func_timeout import func_timeout, FunctionTimedOut
import numpy as np

from bs4 import BeautifulSoup
from nltk import ngrams

from ...utils import truncate_en_doc, truncate_zh_doc


class SyntaxExtractor:
    def __init__(self, config):
        self.max_doc_len = config["max_doc_len"]
        self.min_doc_len = config["min_doc_len"]
        self.zh_stopwords = ['版权归原作者所有，如有侵权，请联系我们', " 您的浏览器不支持 video 标签", "\r",
                             "特别声明：以上内容(如有图片或视频亦包括在内)为自媒体平台“网易号”用户上传并发布，本平台仅提供信息存储服务。"]
        self.en_stopwords = [
            "Stack Exchange network consists of 183 Q&A communities including Stack Overflow, the largest, most trusted online community for developers to learn, share their knowledge, and build their careers.",
            "Do Not Sell My Personal Information",
            "The technical storage or access that is used exclusively for anonymous statistical purposes.",
            "Without a subpoena, voluntary compliance on the part of your Internet Service Provider, or additional records from a third party, information stored or retrieved for this purpose alone cannot usually be used to identify you.",
            "All rights reserved.",
            "Reddit, Inc. © 2023.",
            "We use cookies to help us to deliver our services. We'll assume you're ok with this, but you may change your preferences at our Cookie Centre.",
        ]

        self.extract = lambda lang, html, loosen=False: self.zh_extract(html, loosen) if lang == "zh" else self.en_extractor(html, loosen)

    def zh_extract(self, html, loosen=False):
        soup = BeautifulSoup(html, 'lxml')
        a = soup.get_text()
        b = a.replace(" ", " ").replace("​", " ").replace("﻿", " ")
        c = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                   " ", b)
        c = emoji.replace_emoji(c, replace='')
        for i in self.zh_stopwords:
            c = c.replace(i, "")

        raw = c.split("\n")

        filter_ = set()
        paragraphs = []
        
        char_min_num = 15 if not loosen else 5
        punctuation_min_num = 2 if not loosen else 0

        for index, text in enumerate(raw):
            if 1 < index < len(raw) - 1 and len(raw[index - 1].strip()) >= char_min_num and len(raw[index + 1].strip()) >= char_min_num and \
                    sum([text.count(i) for i in "，。？；！"]) > punctuation_min_num:
                paragraphs.append(text.strip())
                continue

            if len(text.strip()) < char_min_num:
                filter_.add(text)
                continue

            if sum([text.count(i) for i in "，。？；"]) <= punctuation_min_num:
                filter_.add(text)
                continue

            paragraphs.append(text.strip())

        doc = "\n".join(paragraphs)

        doc = truncate_zh_doc(doc, self.max_doc_len, self.min_doc_len)

        return doc

    def en_extractor(self, html, loosen=False):
        try:
            t = func_timeout(15, self.warp_en_extractor, args=(html, loosen))
        except FunctionTimedOut:
            logger.info('整体去重函数超时')
            t = ""
        return t

    def warp_en_extractor(self, html, loosen=False):
        soup = BeautifulSoup(html, 'lxml')
        text = soup.get_text()
        a = re.sub(r'[^\S\n\r]+', ' ', text)
        b = a.replace(" ", " ").replace("​", " ").replace("﻿", " ").replace("‍", " ")
        c = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                   " ", b)
        c = emoji.replace_emoji(c, replace='')

        for i in self.en_stopwords:
            c = c.replace(i, "")

        if "Access to this page has been denied" in c:
            return None
        
        if np.mean([len(s.split()) for s in re.split(r"([.!?;,])", c)]) < 1.5 and len(c.split()) > 30:
            return None
        
        raw = c.split("\n")

        filter_ = set()
        paragraphs = []
        
        char_min_num = 10 if not loosen else 5
        punctuation_min_num = 2 if not loosen else 0
        
        for index, text in enumerate(raw):
            if text.count("�") > 1 or len(text.strip()) < char_min_num or sum([text.count(i) for i in ",.?;!"]) < punctuation_min_num:
                filter_.add(text)
                continue

            paragraphs.append(text.strip())

        

        doc = "\n".join(paragraphs)

        doc = re.sub(r"[ ]{5,}", "", doc)

        sentence_list = re.split(r"([.!?;,])", doc)
        sentence_list.append("")
        # concatenate sentence and splitted punctuation
        sentence_list = ["".join(i) for i in zip(sentence_list[0::2], sentence_list[1::2])]

        clean_sentence_list = []
        for sentence in sentence_list:
            flag = True

            if len(sentence.split(" ")) > 100:
                flag = False

            for word in sentence.split(" "):
                if len(word) > 20:
                    flag = False
                    break

            if flag:
                clean_sentence_list.append(sentence)
        sentence_list = clean_sentence_list

        punctuations = [".", ",", "!", "'", '"', "\n"]

        # jaccard similarity
        grams_list = []
        for sentence in sentence_list:
            for p in punctuations:
                sentence = sentence.replace(p, "")
            grams_list.append(set(ngrams(sentence.split(" "), 10)))

        index1 = 0
        index2 = 0
        discard_index = []

        while index1 < len(sentence_list):
            if len(grams_list[index1]) == 0 or index1 in discard_index:
                index1 += 1
                continue

            index2 = index1 + 1
            while index2 < len(sentence_list):
                if len(grams_list[index2]) == 0 or index2 in discard_index:
                    index2 += 1
                    continue

                similarity = len(grams_list[index1] & grams_list[index2]) / len(grams_list[index1] | grams_list[index2])
                if similarity > 0.2:
                    if len(grams_list[index1]) > len(grams_list[index2]):
                        discard_index.append(index2)
                    else:
                        discard_index.append(index1)
                        break

                index2 += 1

            index1 += 1

        doc_list = []
        for index, sentence in enumerate(sentence_list):
            if index not in discard_index:
                doc_list.append(sentence.strip())
        doc = " ".join(doc_list)

        # one word is 2~3 tokens
        # if self.min_doc_len is not None and len(doc.split(" ")) * 2 <= self.min_doc_len:
        #     return None

        # if self.max_doc_len is not None:
        #     doc = " ".join(doc.split(" ")[:self.max_doc_len // 2])

        #     index = len(doc) - 1
        #     while index >= 0:
        #         if doc[index] in ".!?\n":
        #             return doc[:index + 1]
        #         else:
        #             index -= 1
        doc = truncate_en_doc(doc, self.max_doc_len, self.min_doc_len)

        return doc

