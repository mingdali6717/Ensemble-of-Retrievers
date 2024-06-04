# Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model
This is the official repository for our ACL 2024 (findings) paper [Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model](https://arxiv.org/abs/2405.20680). We propose Ensemble of Retrievers (EoR), a trainable generate-then-rerank framework that first generates answers from different retrievers (we implement 16 different retrievers including retreival-free) and then select the best one out of them. Our experiments demonstrate that EoR can adaptively retrieve from different knowledge sources and boost the performance of Retrieval-Augmented Language Models.

Our framework is composed of three parts: **Controller**, **Generator** and **Evaluator**. The Controller takes the responsibility to retrieve and process differet knowledge (i.e. implementing different retrievers) and the Generator generate responses for each retriever. Then the Evaluator evaluate all responses and select the best one as the final answer. 

The **Controller** consists of five different modules, three for retrieving (**Web**, **Wiki**, **Gendoc**) and two for knowledge processing (**Contriever** and **Summarizer**). The **Evaluator** consisted of two different modules: **VoterModule** and **ScorerModule**.
- **Web**: retrieve documents from the search engine. We use [serper api](https://serper.dev/) for web search engine.
- **Wiki**: we retrieve documents from English Wikipedia dump from Dec. 20, 2018.
- **Gendoc**: we follow the paper [Gendoc](https://arxiv.org/abs/2209.10063) to generate documents for LLM.
- **Contriever**: chunk the documents and then use a dense retrieval model to reranking the chunks.
- **Summarizer**: summarize the given documents by LLM.
- **VoterModule**: score each response by comparing its similarity to other responses.
- **ScorerModule**: score each response by a rewarding model. (deactivated in our paper for short-answer QA. but we find it effective for long-form QA)

Please note that the Web module conducts on-time search-engine retrieval. Hence the results might be different from our own experiments presented in the paper. We will release our intermediate results in the future.



# Experiments
1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Config](#Config)
4. [Run EoR](#EoR)
5. [Other](#Other)

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
  - Bem model (Used for evaluation). Please download the model from [Download link](https://tfhub.dev/google/answer_equivalence/bem/1) then save it in **data/model/** directory (or change the default cache path **CACHED_BEM_PATH** in [eor/evaluation/\__init__.py](https://github.com/mingdali6717/Ensemble-of-Retrievers/blob/master/eor/evaluation/__init__.py) line 40). 
  - webglm_dual_encoder (Used for reranking in Contriever Module, from [WebGLM](https://github.com/THUDM/WebGLM). Here's the [download link](https://cloud.tsinghua.edu.cn/d/bc96946dd9a14c84b8d4/). Then please change the corresponding model_path in the config file to your cached path as follows:
    ```bash
    yaml Config file
     └───LLMConfig
         └───llm5
             │───model_name: en_paragraph_encoder
             │───model_path: **change to your cached path here**
    ```
- Other models needed can be downloaded directly from huggingface. you can add or replace any model by modifying the config file as follows:
  - step 1, modify LLMConfig to add a new model to the framework
    ```bash
    yaml Config file
     └───LLMConfig
         └───[anything is ok, not important]
             │───model_name: [the name to represent this model]
             │───model_path: [path used for loading the model using .from_pretrained in huggingface]
             │───model_class: [the model class used before .from_pretrained in huggingface, such as 'AutoModel']
             │───fp16: [Whether use half precision]
             │───tokenizer_path: [path used for loading the tokenizer using .from_pretrained in huggingface]
             │───tokenizer_class: [tokenizer class used before .from_pretrained in huggingface]
    ```
  - step 2, change the model name in ModuleConfig to activate the model for the corresponding module.
    ```bash
    yaml Config file
     └───ModuleConfig
         └───[the module name you want to change the base model used]
             │───en_model_name: [put the "model name" in LLMConfig here]
             
    ```
- Reward Model. This framwork also supports to add an rewarding model to score each response after voting (not clearly illustrated in our papers). You can train your own rewarding model by  training daryl149/llama-2-7b-chat-hf with openai/webgpt_comparisons and DeepSpeed-Chat Framwork. [Raw training code link](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning) or use codes in our project（under **train_reward_model** directory）
  



## Datasets
Download the datasets NQ, TriviaQ, WebQ from this [url](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f), then run the following code to process the data:
```
python data/datasets/process.py -d [path to the data jsonl file] -o [path to save the processed data]
```
to sample the 500 examples for gpt-3.5, run the following code:
```
python data/datasets/data_sampler.py -d [path to the processed data jsonl file] -o [path to save the sampled data]
```
## Config
we use yaml config file to control the parameters in our framework. you can find them in 'config/'.\

you can add or modify the module config under "ModuleConfig" to change the basic configuration for the seven basic modules

'LLMconfig' controls the loading configurations for the models used in this framework.

'ControlConfig' controls the configuration of different retrievers.

'ControllerConfig','GeneratorConfig' and 'EvaluatorConfig' controlls what modules to be used.


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
for quick search, please cache all intermediate results of each module when generating answers. 

Notebly, you can find a file named "controller_result.json" in your cached result directory which cached all processed documents. You can specify this file's name to skip all modules in the Controller.
```
python run_search_weight.py -c [path of config file] -d [name of dataset to evaluate] -m [model name] -r [path of cached result dir] -t [path of saved query for train data]  -cn [file name of the cached controller result file] --test_path [path of saved query for test data] --test_cached_path [path of cached test result dir] -o [path to save parameter search result] 
```
## Other
- We use [serper api](https://serper.dev/) for web search engine. If you want to use web search module, please change the serper api-key to your own in [eor/retrieval/web/search.py](https://github.com/mingdali6717/Ensemble-of-Retrievers/blob/master/eor/retrieval/web/search.py) line 6.
- If you want to use OpenAI models, please add your OpenAI keys to [eor/utils/openai_tools.py](https://github.com/mingdali6717/Ensemble-of-Retrievers/blob/master/eor/utils/openai_tools.py) line 13 'API_KEYS'.
- You can modify the config file according to your need in the yaml file in **Config** directory, please refer to our given example.



