import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer)

import modules.shared as shared



def find_model_type(model_name):
    path_to_model = Path(f'{shared.model_dir}/{model_name}')
    if not path_to_model.exists():
        return 'None'

    model_name_lower = model_name.lower()
    
    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.trust_remote_code)
    # Not a "catch all", but fairly accurate
    if config.to_dict().get("is_encoder_decoder", False):
        return 'HF_seq2seq'
    else:
        return 'HF_generic'
    

def load_model(model_name, gptq = False):
    print(f"Loading {model_name}...")
    t0 = time.time()

    shared.model_type = find_model_type(model_name)
    if shared.model_type == 'None':
        print('The path to the model does not exist. Exiting.')
        return None, None
    
    if gptq == True:
        load_func = AutoGPTQ_loader
    else:
        load_func = huggingface_loader

    output = load_func(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)



    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.\n")
    return model, tokenizer

def load_tokenizer(model_name, model):
    tokenizer = None

    if type(model) is transformers.LlamaForCausalLM or "LlamaGPTQForCausalLM" in str(type(model)):
        # Try to load an universal LLaMA tokenizer
        for p in [Path(f"{shared.model_dir}/llama-tokenizer/"), Path(f"{shared.model_dir}/oobabooga_llama-tokenizer/")]:
            if p.exists():
                print(f"Loading the universal LLaMA tokenizer from {p}...")
                tokenizer = LlamaTokenizer.from_pretrained(p, clean_up_tokenization_spaces=True)
                return tokenizer

        # Otherwise, load it from the model folder and hope that these
        # are not outdated tokenizer files.
        tokenizer = LlamaTokenizer.from_pretrained(Path(f"{shared.model_dir}/{model_name}/"), clean_up_tokenization_spaces=True)
        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass
    else:
        path_to_model = Path(f"{shared.model_dir}/{model_name}/")
        if path_to_model.exists():
            tokenizer = AutoTokenizer.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

    return tokenizer


def huggingface_loader(model_name):
    if shared.model_type == 'chatglm':
        LoaderClass = AutoModel
    elif shared.model_type == 'HF_seq2seq':
        LoaderClass = AutoModelForSeq2SeqLM
    else:
        LoaderClass = AutoModelForCausalLM

    # Load the model in simple 16-bit mode by default
    model = LoaderClass.from_pretrained(Path(f"{shared.model_dir}/{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=shared.trust_remote_code)
    
    model = model.cuda()
  
    return model

def GPTQ_loader(model_name):
    import modules.GPTQ_loader

    model = modules.GPTQ_loader.load_quantized(model_name)

    return model

def AutoGPTQ_loader(model_name):
    import modules.AutoGPTQ_loader

    return modules.AutoGPTQ_loader.load_quantized(model_name)


def get_max_memory_dict():
    max_memory = {}

    total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    suggestion = round((total_mem - 1000) / 1000) * 1000
    if total_mem - suggestion < 800:
        suggestion -= 1000

    suggestion = int(round(suggestion / 1000))
    print(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
    max_memory = {0: f'{suggestion}GiB', 'cpu': f'{64}GiB'}

    return max_memory if len(max_memory) > 0 else None