from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import math
import re

def truncate_en_doc(doc: str, max_doc_len: int = 1000, min_doc_len: int = None, tokenizer=None, keep_sentence=True):
    """
    truncate doc if doc_len >= max_length, doc will be truncated and discard last uncompleted sentence. if doc_len <= min_length, return will be None. doc_len is defined len(doc.split(" "))

    Args:
        doc (str): _description_
        max_doc_len (int, optional): _description_. Defaults to None.
        min_doc_len (int, optional): _description_. Defaults to None.
    """
    if tokenizer is None:
        words = doc.split(" ")
        word_len = len(words)
    
    elif tokenizer == "nltk":
        try:
            words = [word_tokenize(d) for d in doc.split("\n")]
        except:
            import nltk
            nltk.download('punkt')
            words = [word_tokenize(d) for d in doc.split("\n")]
        
        chunk_lens = [len(w) for w in words]
        word_len = sum(chunk_lens)
    elif getattr(tokenizer, 'encode', None) is not None and getattr(tokenizer, 'decode', None) is not None:
        #tokenr = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")
        words = tokenizer.encode(doc, add_special_tokens=False)
        word_len = len(words)
    else:
        raise KeyError("tokenizer should be one of None, llama, nltk.")
    
    

    if min_doc_len is not None and word_len <= min_doc_len:
        return None

    if max_doc_len is None or word_len <= max_doc_len:
        return doc
    
    if tokenizer == 'nltk':
        total_sum = 0
        chunk_id = -1

        while total_sum <= max_doc_len:
            chunk_id += 1
            total_sum += chunk_lens[chunk_id]
            

        chunk_idx = sum(chunk_lens[:chunk_id+1]) - max_doc_len
        prefix_doc_list = words[:chunk_id] + [words[chunk_id][:-chunk_idx]]
        doc = "\n".join([TreebankWordDetokenizer().detokenize(w) for w in prefix_doc_list])
    elif tokenizer is None:
        doc = " ".join(words[:max_doc_len])
    else:
        doc = tokenizer.decode(words[:max_doc_len], skip_special_tokens=True)
        
    if keep_sentence:
        index = len(doc) - 1
        while index >= 0:
            if doc[index] in ";,.!?\n":
                doc = doc[:index + 1]
                break
            else:
                index -= 1

    return doc


def truncate_zh_doc(doc: str, max_doc_len: int = None, min_doc_len: int = None, tokenizer=None, keep_sentence=True):
    
    
    if tokenizer is None or tokenizer == 'nltk':
        
        word_len = len(doc)
    elif getattr(tokenizer, 'encode', None) is not None and getattr(tokenizer, 'decode', None) is not None:
    
        words = tokenizer.encode(doc, add_special_tokens=False)
        word_len = len(words)
    else:
        raise KeyError("tokenizer should be one of None, llama, nltk.")
    
    if min_doc_len is not None and word_len <= min_doc_len:
        return None

    if max_doc_len is None or word_len <= max_doc_len:
        return doc
    
    if tokenizer is None or tokenizer == 'nltk':
        doc = doc[:max_doc_len]
    else: 
        doc = tokenizer.decode(words[:max_doc_len], skip_special_tokens=True)
    if keep_sentence:
        index = len(doc) - 1
        while index >= 0:
            if doc[index] in "。，；！？.?!;,\n":
                doc = doc[:index + 1]
                break
            else:
                index -= 1

    return doc

def split_with_fix_length_en(doc:str, fix_length: int = 100, keep_sentence: bool = True, keep_last: bool = True, min_last_length: int = None, tokenizer=None, max_chunk_length=None):
    """
    given a doc, split into chunks, each chunk with minimum length {fix_length}.

    Parameters:
    doc: str - doc to be chunked
    fix_length: int - minimum length to chunk, length depend on the tokenizer. if keep_sentence is false, each chunk exactly have length {fix length}
    keep_sentence: bool - if True, last sentence in each chunk will be complemented. truncated at the {max_chunk_length}
    max_chunk_length: int - if keep_sentence is True, each chunk length is between  {fix_length, max_chunk_length}
    keep_last: bool -  if False, the last chunk with length less than fix_length in the doc will be throwed away.
    min_last_length: if last chunk length is less than min_last_length, it will be appended to the tail of last chunk.
    """
    if max_chunk_length is None and keep_sentence:
        max_chunk_length = math.ceil(fix_length * 1.1)
    if keep_last and min_last_length is None:
        min_last_length = math.ceil(fix_length * 1.1)

    if tokenizer is None:
        words = doc.split(" ")
        stop_words = ".!?;\n"
    elif tokenizer == 'nltk':
        try:
            words = word_tokenize(doc)
        except:
            import nltk
            nltk.download('punkt')
            words = word_tokenize(doc)
        stop_words = ".!?;\n"
    
    else:
        raise KeyError("tokenizer should be one of None, nltk.")
    
    if len(words) <= fix_length:
        return [doc]
    
    splited_doc = []

    if keep_sentence:
        while len(words) > fix_length:
            index = fix_length - 1
            if tokenizer is None:
                while re.search(r"\b[.;!?\n]", words[index]) is None and words[index] not in stop_words and index < len(words)-1:
                    index += 1
            else:
                while words[index] not in stop_words and index < len(words)-1:
                    index += 1
            if index > max_chunk_length:
                splited_doc.append(words[:max_chunk_length])
            else:
                splited_doc.append(words[:index+1])

            words = words[index+1:]
    else:
        while len(words) > fix_length:
            splited_doc.append(words[:fix_length])
            words = words[fix_length:]

    if keep_last:
        if len(words) > min_last_length:
            splited_doc.append(words)
        else:
            splited_doc[-1].extend(words)

    if tokenizer is None: 
        return [" ".join(s) for s in splited_doc]
    elif tokenizer == 'nltk':
        return [TreebankWordDetokenizer().detokenize(s)  for s in splited_doc]
    
def split_with_fix_length_zh(doc:str, fix_length: int = 100, keep_sentence: bool = True, keep_last: bool = True, min_last_length: int = 15, tokenizer=None, max_chunk_length=None):
    
    if max_chunk_length is None and keep_sentence:
        max_chunk_length = math.ceil(fix_length * 1.1)
    if keep_last and min_last_length is None:
        min_last_length = math.ceil(fix_length * 1.1)

    words = doc

    if len(words) <= fix_length:
        return [words]

    splited_doc = []
    
    if keep_sentence:
    
        while len(words) > fix_length:
            index = fix_length - 1
            while words[index] not in "。！？；.!?;\n" and index < len(words):
                index += 1
            if index > max_chunk_length:
                splited_doc.append(words[:max_chunk_length])
            else:
                splited_doc.append(words[:index+1])

            words = words[index+1:]
    else:
        while len(words) > fix_length:
            splited_doc.append(words[:fix_length])
            words = words[fix_length:]
    
    if keep_last:
        if len(words) > min_last_length:
            splited_doc.append(words)
        else:
            splited_doc[-1].extend(words)
    
    return splited_doc

def count_text_len(doc, tokenizer=None):

    try:
        words = word_tokenize(doc)
    except:
        import nltk
        nltk.download('punkt')
        words = word_tokenize(doc)

    return len(words)