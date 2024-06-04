# Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model
This is the official repository for our ACL 2024 (findings) paper [Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model](https://arxiv.org/abs/2405.20680).



# Experiments
1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Run EoR](#EoR)
4. [Other](#Other)

## Installation
1. Create python environment
    ```bash
    conda create -n EoR python=3.9
    ```
2. Install pytorch
    ```bash
    pip3 install torch
    ```
3. 
    ```bash
    conda install -c conda-forge openjdk=11
    ```
4. 
    ```bash
    conda install -c pytorch faiss-cpu=1.7.4 mkl=2021
    ```
5. 
    ```bash
    pip install -r requirements.txt
    ```

**Other models and files needed:**

  -	Wiki index files. After installing pyserini, using codes below to download.（About 50G. please save the proxy.Index files in ~/.cache/pyserini/indexes）
    ```python
    from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
    en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))
    ```
  - Bem model (Used for evaluation). Please download the model from [Download link](https://tfhub.dev/google/answer_equivalence/bem/1) then save it in **data/model/** directory (or change the default cache path **CACHED_BEM_PATH** in eor/evaluation/\__init__.py line 40). 
  - webglm_dual_encoder (Used for reranking in Contriever Module, from [WebGLM](https://github.com/THUDM/WebGLM). Here's the [download link](https://cloud.tsinghua.edu.cn/d/bc96946dd9a14c84b8d4/). Then please change the corresponding model_path in the config file to your cached path as follows:
    ```bash
    yaml Config file
     └───LLMConfig
         └───llm5
             │───model_name: en_paragraph_encoder
             │───model_path: **change to your cached path here**
    ```
- Other models needed can be downloaded directly from huggingface. you can add or replace any model by modifying the config file as follows:
  - step 1, modify LLMConfig to add a now model to the framework
    ```bash
    yaml Config file
     └───LLMConfig
         └───[anything is ok, not important]
             │───model_name: [the name to represent this model]
             │───model_path: [path used for loading model by .from_pretrained in huggingface]
             │───model_class: [the class before .from_pretrained in huggingface]
            
    ```
  - Reward Model.Training daryl149/llama-2-7b-chat-hf with openai/webgpt_comparisons and DeepSpeed-Chat Framwork.[Raw training code link](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning) or use codes in our project（under **train_reward_model** directory）
  



## Datasets
Download the datasets NQ, TriviaQ, WebQ from this [url](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f), then run the following code to process the data:
```
python data/datasets/process.py -d [path to the data jsonl file] -o [path to save the processed data]
```
to sample the 500 examples for gpt-3.5, run the following code:
```
python data/datasets/data_sampler.py -d [path to the processed data jsonl file] -o [path to save the sampled data]
```
## EoR
### Generate Answers
```
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py -c [path to the config file]
```

- HF_DATASETS_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are not necessary.
### Evaluation
```
python run_evaluate.py -r [path to the result json file ] -o [path to save evaluation result] -n [dataset_name] -m [metrics to evaluate]
```
### Parameter Search
```
python run_search_weight.py -c [path of config file] -d [name of dataset to evaluate] -m [model name] -r [path of cached result dir] -t [path of saved query for train data]  -cn [file name of the cached controller result file] --test_path [path of saved query for test data] --test_cached_path [path of cached test result dir] -o [path to save parameter search result] 
```
## Other
- We use [serper api](https://serper.dev/) for web search engine.If use web search module,please change the serper api-key of your own in line12 of [eor/retrieval/web/search.py](https://github.com/mingdali6717/Ensemble-of-Retrievers/blob/master/eor/retrieval/web/search.py)
- You can write or modify config file according to your need in yaml file in **Config** director, please refer to our given example.



