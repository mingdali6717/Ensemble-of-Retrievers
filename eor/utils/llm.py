from typing import List, Optional
import torch
import math
from tqdm import tqdm
import transformers
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader
from .openai_tools import prompt_openai_api, OPENAI_MODEL_LIST, reshape_sequences
from .modeling_reward_model import RewardModel
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger



MAX_LENGTH = int(10000)

class LLM:
    """"
    除非使用OPENAI API，否则lm_{generate, encode, reward}均需要传入gpu，没有默认。
    DataParallel这种一个进程多个GPU的，慢且复杂，建议DistributeDataParallel、torch.multiprocessing或Deepspeed
    """
    llm_config_dict = {}
    llms = {}
    openai_usage_log = None
    gpu_ids = []
    
    @classmethod
    def get_llm_config(cls, _config):
        for value in _config.values():
            cls.llm_config_dict[value["model_name"]] = value

    @classmethod
    def initial_all(cls, device_name, llm_names):
        if "gpu" in device_name:
            device = torch.device(f"cuda:{device_name.replace('gpu','')}")
        else:
            device = torch.device("cpu")
        for llm_name in llm_names.replace(" ","").split(","):
            llm_config = cls.llm_config_dict[llm_name]
            
            if "reward_model" not in llm_name:
                model_class = getattr(transformers, llm_config["model_class"])
                model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True, 
                                                    torch_dtype=torch.float16 if llm_config["fp16"] else torch.float32,
                                                    device_map=device)
                if cls.ddp:
                    model = DDP(model, device_ids=[int(device_name.replace("gpu", ""))])
                
                model.eval()
                    
                tokenizer_class = getattr(transformers, llm_config["tokenizer_class"])
                tokenizer_path = llm_config.get("tokenizer_path", llm_config["model_path"])
                tokenizer = tokenizer_class.from_pretrained(tokenizer_path, trust_remote_code=True)
                
                if "llama2" in llm_name:
                    tokenizer.pad_token = tokenizer.unk_token
                
                cls.llms[device_name + ":" + llm_config["model_name"]] = (model, tokenizer)
                
                print(f"Successfully initial {device_name + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")
            else:
                if "llama2" in llm_config["model_name"]:
                    base_model = LlamaModel.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
                    tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", fast_tokenizer=True)

                    tokenizer.pad_token = tokenizer.eos_token
                    base_model.config.end_token_id = tokenizer.eos_token_id
                    base_model.config.pad_token_id = tokenizer.pad_token_id

                    base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
                    
                    reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)
                    
                    reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')))
                    
                    if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                        reward_model.half()
                    
                    reward_model.to(device)
                    
                    if cls.ddp:
                        reward_model = DDP(reward_model, device_ids=[int(device_name.replace("gpu", ""))])
                    
                    reward_model.eval()
                    
                    cls.llms[device_name + ":" + llm_config["model_name"]] = (reward_model, tokenizer)
                    
                    print(f"Successfully initial {device_name + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")
                elif "xverse" in llm_config["model_name"]:
                    base_model = LlamaModel.from_pretrained("xverse/XVERSE-13B", trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-13B", fast_tokenizer=True)

                    tokenizer.pad_token = tokenizer.eos_token
                    base_model.config.end_token_id = tokenizer.eos_token_id
                    base_model.config.pad_token_id = base_model.config.eos_token_id

                    base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
                    
                    reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

                    reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')))

                    if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                        reward_model.half()
                    
                    reward_model.to(device)
                    
                    if cls.ddp:
                        reward_model = DDP(reward_model, device_ids=[int(device_name.replace("gpu", ""))])
                    
                    reward_model.eval()
                    
                    cls.llms[device_name + ":" + llm_config["model_name"]] = (reward_model, tokenizer)
                    
                    print(f"Successfully initial {device_name + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")
    
    @classmethod
    def initial_lm(cls, model_name, device_name, verbose=True):
        """
        model_name: should be one of supported model in LLMConfig model_name.
        device_name: in the format of f'gpu{device_id}' or 'cpu'
        """
        if "gpu" in device_name:
            device = torch.device(f"cuda:{device_name.replace('gpu','')}")
        else:
            device = torch.device("cpu")

        

        if device_name + ":" + model_name in cls.llms.keys():
            if verbose:
                logger.info(f"***************{device_name}: REUSE LOADED MODEL '{device_name+':'+model_name}*************")
            return device_name + ":" + model_name
        else: 
            if verbose:
                logger.info(f"***************{device_name}: LOAD MODEL '{device_name+':'+model_name} FROM SCRATCH*************")
    
        
        llm_config = cls.llm_config_dict[model_name]
        
    
        model_class = getattr(transformers, llm_config["model_class"])
        model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True, 
                                            torch_dtype=torch.float16 if llm_config["fp16"] else torch.float32,
                                            device_map=device)
        
        if cls.ddp:
            model = DDP(model, device_ids=[int(device_name.replace("gpu", ""))])
            if verbose:
                logger.info(f"***************DDP model setting finished for {device_name}*************")
        
        model.eval()
        
        tokenizer_class = getattr(transformers, llm_config["tokenizer_class"])
        tokenizer_path = llm_config.get("tokenizer_path", llm_config["model_path"])
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        if ("baichuan" in tokenizer.name_or_path or "llama" in tokenizer.name_or_path or "Llama" in tokenizer.name_or_path) and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        cls.llms[device_name + ":" + model_name] = (model, tokenizer)
        
        print(f"Successfully initial {device_name + ':' + model_name}")
        
        return device_name + ":" + model_name
    
    @classmethod
    def initial_rm(cls, model_name, device_name):
        if device_name + ":" + model_name in cls.llms.keys():
            return device_name + ":" + model_name
        
        if "gpu" in device_name:
            device = torch.device(f"cuda:{device_name.replace('gpu','')}") 
        else: 
            device = torch.device("cpu")
        llm_config = cls.llm_config_dict[model_name]
        
        if "llama2" in model_name:
            base_model = LlamaModel.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
            tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", fast_tokenizer=True)

            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.end_token_id = tokenizer.eos_token_id
            base_model.config.pad_token_id = tokenizer.pad_token_id

            base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
            
            reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)
            
            reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')))

            if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                reward_model.half()

            reward_model.to(device)
            
            if cls.ddp:
                reward_model = DDP(reward_model, device_ids=[int(device_name.replace("gpu", ""))])
            
            reward_model.eval()
            
            cls.llms[device_name + ":" + model_name] = (reward_model, tokenizer)
        elif "xverse" in model_name:
            base_model = LlamaModel.from_pretrained("xverse/XVERSE-13B", trust_remote_code=True)
            tokenizer = LlamaTokenizer.from_pretrained("xverse/XVERSE-13B", fast_tokenizer=True)

            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.end_token_id = tokenizer.eos_token_id
            base_model.config.pad_token_id = base_model.config.eos_token_id

            base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
            
            reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

            reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')))
            
            if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                reward_model.half()
                
            reward_model.to(device)
            
            if cls.ddp:
                reward_model = DDP(reward_model, device_ids=[int(device_name.replace("gpu", ""))])
            
            reward_model.eval()
            
            cls.llms[device_name + ":" + model_name] = (reward_model, tokenizer)

        print(f"Successfully initial {device_name + ':' + model_name}")
        
        return device_name + ":" + model_name
    
    @classmethod
    def release_one(cls, model_name):
        del cls.llms[model_name][0]
        del cls.llms[model_name][1]
    
    @classmethod
    def release_all(cls):
        for llm in cls.llms.values():
            model, tokenizer = llm
            del model
            del tokenizer
        
        cls.llms = {}
        
        if cls.openai_usage_log is not None:
            cls.openai_usage_log.close()        
    
    @classmethod
    def lm_generate(cls, **kwargs):
        if kwargs["model_name"] in OPENAI_MODEL_LIST:
            
            if "max_new_tokens" in kwargs["generate_kwargs"]:
                if kwargs["generate_kwargs"]["max_new_tokens"] is not None:
                    kwargs["generate_kwargs"]["max_tokens"] = kwargs["generate_kwargs"]["max_new_tokens"]
                kwargs["generate_kwargs"].pop("max_new_tokens")
            
            if "num_responses_per_prompt" in kwargs["generate_kwargs"]:
                kwargs["generate_kwargs"]["n"] = kwargs["generate_kwargs"]["num_responses_per_prompt"]
                kwargs["generate_kwargs"].pop("num_responses_per_prompt")

            generated_sequences, _ = prompt_openai_api(kwargs["model_name"], kwargs["prompts"], **kwargs["generate_kwargs"])
            return generated_sequences
        else:
            model_name = cls.initial_lm(kwargs["model_name"], kwargs["device_name"], kwargs["verbose"])
            model, tokenizer = cls.llms[model_name]
            if ("num_responses_per_prompt" in kwargs["generate_kwargs"]) :
                if kwargs["generate_kwargs"]["do_sample"] and (kwargs["generate_kwargs"]["num_responses_per_prompt"] > 1):
                    n = kwargs["generate_kwargs"]["num_responses_per_prompt"]
                else:
                    if kwargs["generate_kwargs"]["num_responses_per_prompt"] > 1:
                        logger.warning(f"num_responses_per_prompt is set to {kwargs['generate_kwargs']['num_responses_per_prompt']} > 1, but do_sample is set to False, hence only 1 response will be generated for each prompt")
                    n = 1
                kwargs["generate_kwargs"].pop("num_responses_per_prompt")
            else:
                n = 1

            if "truncation_side" in kwargs["tokenize_kwargs"]:
                if kwargs["tokenize_kwargs"]["truncation_side"] is not None:
                    assert kwargs["tokenize_kwargs"]["truncation_side"] in ["left", "right"], f"truncation side should be 'left' or 'right', but {kwargs['tokenize_kwargs']['truncation_side']} is given"
                    tokenizer.truncation_side = kwargs["tokenize_kwargs"]["truncation_side"]
                    kwargs["tokenize_kwargs"].pop("truncation_side")
    
            if "padding_side" in kwargs["tokenize_kwargs"]:
                if kwargs["tokenize_kwargs"]["padding_side"] is not None:
                    assert kwargs["tokenize_kwargs"]["padding_side"] in ["left", "right"], f"padding side should be 'left' or 'right', but {kwargs['tokenize_kwargs']['padding_side']} is given"
                    tokenizer.padding_side = kwargs["tokenize_kwargs"]["padding_side"]
                    kwargs["tokenize_kwargs"].pop("padding_side")
            

            generated_sequences = cls._frozen_lm_generate(model, tokenizer, kwargs["prompts"], kwargs["tokenize_kwargs"], kwargs["generate_kwargs"], n=n)

            return generated_sequences
    
    @classmethod
    def _frozen_lm_generate(cls, model, tokenizer, prompts, tokenize_kwargs, generate_kwargs, n=1) -> List[str]:
        
        if type(prompts) is str:
            prompts = [prompts]
        
        if n > 1:
            prompts = [p for p in prompts for k in range(n)]

        
        module = model.module if LLM.ddp else model
        max_sequence_length = max(getattr(module.config, "max_position_embeddings", 0), getattr(module.config, "n_positions", 0), getattr(module.config, "seq_length", 0))
        max_prompt_length, max_new_tokens = adjust_length_to_model(generate_kwargs["max_new_tokens"], max_sequence_length) 
        generate_kwargs["max_new_tokens"] = max_new_tokens
        tokenize_kwargs["max_length"] = max_prompt_length
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["return_tensors"] = "pt"
        
        tokenize_prompt = tokenizer(prompts, **tokenize_kwargs)
        
        batch_size = generate_kwargs.pop("batch_size", 10)
        
        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids = tokenize_prompt["input_ids"].to(module.device)
                attention_mask = tokenize_prompt["attention_mask"].to(module.device)
                output_sequences = module.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

                output_seq = output_sequences[:, input_ids.shape[1]:]
                output_str = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if n > 1:
                    output_str = reshape_sequences(output_str, n)
                
                return output_str
            else:
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if module.dtype==torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokenize_prompt.data), batch_size=batch_size, collate_fn=data_collator)
                generated_sequences = []
                
                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.device)
                    attention_mask = batch["attention_mask"].to(module.device)
                    output_sequences = module.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
                    # print_gpu_usage(input_ids.get_device())

                    output_seq = output_sequences[:, input_ids.shape[1]:]
                    output_str = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    generated_sequences.extend(output_str)
                    
                if n > 1:
                    generated_sequences = reshape_sequences(generated_sequences, n)

                return generated_sequences
    
    
    @classmethod
    def lm_encode(cls, model_name, prompts, device, tokenize_kwargs, encode_kwargs):
        if type(prompts) is str:
            prompts = [prompts]
        
        model_name = cls.initial_lm(model_name, device)

        model, tokenizer = cls.llms[model_name]
        
        tokenize_kwargs["return_tensors"] = "pt"
        tokens = tokenizer(prompts, **tokenize_kwargs)
        
        batch_size = encode_kwargs.get("batch_size", 10)
        pooling_method = encode_kwargs.get("pooling_method", None)
        
        assert pooling_method in ["mean", "sum", "max", "cls"] if pooling_method is not None else True
        
        module = model.module if cls.ddp else model
        
        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids, attention_mask = tokens["input_ids"].to(module.device), tokens["attention_mask"].to(module.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                output = LLM.pooling(output, pooling_method, attention_mask).cpu()
            else:
                outputs = [] 
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if module.dtype==torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokens.data), batch_size=batch_size, collate_fn=data_collator)
                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.device)
                    attention_mask = batch["attention_mask"].to(module.device)
                    output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    output = LLM.pooling(output, pooling_method, attention_mask).cpu()
                    outputs.append(output)
                output = torch.cat(outputs)
        return output

    @staticmethod
    def pooling(output, pooling_method, attention_mask=None):
        if pooling_method is None:
            return output
        elif pooling_method == "cls":
            return output[:, 0, :]
        
        assert attention_mask is not None, "For pooling_method in [mean, sum, max], attention_mask is needed."
        
        no_padding_output = output.masked_fill(~attention_mask[..., None].bool(), 0.)

        if pooling_method == "mean":
            output = no_padding_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif pooling_method == "sum":
            output = no_padding_output.sum(dim=1)
        elif pooling_method == "max":
            output = no_padding_output.max(dim=1)
        
        return output
        
    @classmethod
    def lm_reward(cls, model_name, prompts, gpu, tokenize_kwargs, reward_kwargs):
        if type(prompts) is str:
            prompts = [prompts]        
        
        model_name = cls.initial_rm(model_name, gpu)
        
        model, tokenizer = cls.llms[model_name]
        
        tokenize_kwargs["return_tensors"] = "pt"
        tokens = tokenizer(prompts, **tokenize_kwargs)
        
        batch_size = reward_kwargs.get("batch_size", 10)
        
        module = model.module if cls.ddp else model
        
        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids, attention_mask = tokens["input_ids"].to(module.rwtranrsformer.device), tokens["attention_mask"].to(module.rwtranrsformer.device)
                score_list = module.forward_value(input_ids=input_ids, attention_mask=attention_mask, prompt_length=2)
            else:
                score_list = []
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if module.rwtranrsformer.dtype==torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokens.data), batch_size=batch_size, collate_fn=data_collator)
                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.rwtranrsformer.device)
                    attention_mask = batch["attention_mask"].to(module.rwtranrsformer.device)
                    score_list.extend(module.forward_value(input_ids=input_ids, attention_mask=attention_mask, prompt_length=2))
        
        return score_list


def adjust_length_to_model(max_new_tokens, max_sequence_length=0):
    if max_new_tokens < max_sequence_length:
        max_prompt_length = max_sequence_length - max_new_tokens
    elif max_new_tokens >= max_sequence_length and max_sequence_length > 0:
        print(f"model max input length is {max_sequence_length}, but given max_new_tokens are {max_new_tokens}")
        max_prompt_length = int(0.7 * max_sequence_length)
        max_new_tokens = max_sequence_length - max_prompt_length
    else:
        print(f"model max input length is not detected, set max_prompt_length to {MAX_LENGTH}")
        max_prompt_length = MAX_LENGTH

    return max_prompt_length, max_new_tokens
# 
# def print_gpu_usage(rank):
#  
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(rank)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # logger.info("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(rank, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
# 
