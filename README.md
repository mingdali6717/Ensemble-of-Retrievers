# Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model
This is the official repository for our ACL 2024 (findings) paper [Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Model](https://arxiv.org/abs/2405.20680).



# Experiments
1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Run EoR](#EoR)
## Installation

## Datasets
Download the datasets NQ, TriviaQ, WebQ from this [url](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f), then run the following code to process the data:
```
python data/datasets/process.py -d [path to the data jsonl file] -o [path to save the processed data]
```
to sample the 500 examples for gpt-3.5, run the following code:
```
python data/datasets/data_sampler.py -d [path to the processed data jsonl file] -o [path to save the sampled data]
```
## Run EoR
### Generate Answers
### Evaluation
### Parameter Search



