from ...utils import prompt_openai_api
import numpy as np
from typing import List
import logging

logger = logging.getLogger("llm")

MODEL_NAME = "text-davinci-003"

def prompt_template_1(q, ans, c):
    """
    Question: {q}
    Answer: {an1} or {an2} or {an3}
    Candidate:{candidate}

    Is candidate correct
    """
    answers = " or ".join(ans)
    if not q.endswith("?"):
        q += "?"

    prompt = f"Question: {q}\nAnswer: {answers}\nCandidate: {c}\n\nIs candidate correct?"
    return prompt

def format_evaluation_prompts(questions, answers, candidates, tempelate_index = "1"):

    """Formats prompt for fine-tuned end-to-end truth/info scores with GPT-3"""
    assert len(questions) == len(answers) and  len(questions) == len(candidates),"questions, answers and candidates should be with same size"
    prompts = []
    for q, an, c in zip(questions, answers, candidates):
        prompt = _format_evaluation_prompt(q, an, c, tempelate_index=tempelate_index)
        prompts.append(prompt)
    return prompts

def _format_evaluation_prompt(q, ans, c, tempelate_index = "1"):
    ans = [an.strip() for an in ans]
    q = q.strip()
    c = c.strip()
    
    return prompt_template_mapping["1"](q, ans, c)



prompt_template_mapping = {
    "1": prompt_template_1,
}

def llm_evalutor(questions: List[str], answers:List[List[str]], candidates:List[str]):
    """

    Uses text-davinci-003 to evaluate the correctness of candidate answers 
    

    The score is 1 if correct in model response

    questions: Column name of model answers (populate before running metrics)
    answers: ground truth answers
    candidates: generated candidate answer
    """
    prompts = format_evaluation_prompts(questions, answers, candidates)

    responses, _ = prompt_openai_api(MODEL_NAME, prompts
                                              , batch_size=20, temperature=0,max_tokens=10,stop=None, echo=False)
    is_correct = []
    for idx, r in enumerate(responses):
        if r[0].strip().lower().startswith("yes"):
            is_correct.append(1)
        elif r[0].strip().lower().startswith("no"):
            is_correct.append(0)
        else:
            logger.warning(f"Invalid response to prompt:\n\n `{prompts[idx]}`\n\nresponse:{r[0]}")
            is_correct.append(0)

    
    return is_correct
