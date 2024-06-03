import pandas as pd
from statistic_method import em_score
from bem import BemCalculator
import re

bem_scorer = BemCalculator()

def bem_density(row):
    bem_num = 0
    question = row['questions']
    if not isinstance(question, str):
        question = str(question)
    knowledge = row['knowledges']
    if knowledge == ' ':
        return 0
    if not isinstance(knowledge, str):
        knowledge = str(knowledge)
    sentence_list = re.split(r"(?:[.!?;])", knowledge)
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    ground_truths = row['ground_truth']
    if not isinstance(ground_truths, str):
        ground_truths = str(ground_truths)
    ground_truth = ground_truths.split(";")
    for p in sentence_list:
        p = p[:512]
        if not isinstance(p, str):
            p = str(p)
        results = []
        for q in ground_truth:
            if not isinstance(q, str):
                q = str(q)
            example = [{
                'question': question,
                'reference': q,
                'candidate': p
            }]
            result = bem_scorer.bem_score(example)
            results.append(result)
        if max(results) >= 0.8:
            bem_num += 1
    bem_dense = bem_num / len(sentence_list)
    return bem_dense

def em_density(row):
    em_num = 0
    knowledge = row['knowledges']
    if knowledge == ' ':
        return 0
    if not isinstance(knowledge, str):
        knowledge = str(knowledge)
    sentence_list = re.split(r"(?:[.!?;])", knowledge)
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    ground_truths = row['ground_truth']
    if not isinstance(ground_truths, str):
        ground_truths = str(ground_truths)
    ground_truth = ground_truths.split(";")
    for p in sentence_list:
        if not isinstance(p, str):
            p = str(p)
        results = []
        for q in ground_truth:
            if not isinstance(q, str):
                q = str(q)
            result = em_score(p, q)
            results.append(result)
        if max(results) == 1:
            em_num += 1
    em_dense = em_num / len(sentence_list)
    return em_dense

def doc_em(row):
    knowledge = row['knowledges']
    ground_truths = row['ground_truth']
    if not isinstance(ground_truths, str):
        ground_truths = str(ground_truths)
    ground_truth = ground_truths.split(";")
    if not isinstance(knowledge, str):
        knowledge = str(knowledge)
    for p in ground_truth:
        if not isinstance(p, str):
            p = str(p)
        result = em_score(knowledge, p)
        if result == 1:
            return 1
    return 0

def doc_bem(row):
    question = row['questions']
    if not isinstance(question, str):
        question = str(question)
    knowledge = row['knowledges']
    if not isinstance(knowledge, str):
        knowledge = str(knowledge)
    sentence_list = re.split(r"(?:[.!?;])", knowledge)
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    ground_truths = row['ground_truth']
    if not isinstance(ground_truths, str):
        ground_truths = str(ground_truths)
    ground_truth = ground_truths.split(";")
    for p in sentence_list:
        p = p[:512]
        if not isinstance(p, str):
            p = str(p)
        results = []
        for q in ground_truth:
            if not isinstance(q, str):
                q = str(q)
            example = [{
                'question': question,
                'reference': q,
                'candidate': p
            }]
            result = bem_scorer.bem_score(example)
            results.append(result)
        if max(results) >= 0.8:
            return 1
    return 0



# 读取Excel文件
input_file = 'small_doc.xlsx'
df = pd.read_excel(input_file)

try:
    for index, row in df.iterrows():
        df.loc[index, 'doc_em'] = doc_em(row)
        df.loc[index, 'doc_bem'] = doc_bem(row)
        df.loc[index, 'em_density'] = em_density(row)
        df.loc[index, 'bem_density'] = bem_density(row)

    # 保存结果到新的Excel文件
    output_file = 'test_doc.xlsx'
    df.to_excel(output_file, index=False)

    print("处理完成并保存到新的Excel文件.")
except Exception as e:
    print("发生错误:", str(e))
    # 在出现异常时，将已经计算出的部分保存到另一个文件
    error_output_file = 'error_doc.xlsx'
    df.to_excel(error_output_file, index=False)

