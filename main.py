import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


from modules.GPTQ_loader import load_quantized
from modules.text_generation import clear_torch_cache, generate_reply
import modules.shared as shared


shared.model_name = "alpaca-native-4bit"

print(f"Loading {shared.model_name}...")
t0 = time.time()
shared.model = load_quantized(shared.model_name)
shared.tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{shared.model_name}/"))
shared.tokenizer.truncation_side = 'left'


print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")


question = "Explain why Formula 1 is the fastest racing class in the world."
reply = generate_reply(question, max_new_tokens=200, 
               do_sample=False, 
               temperature=0.95, 
               top_p=1, typical_p=1, 
               repetition_penalty=1.1, 
               encoder_repetition_penalty=1, 
               top_k=40, min_length=0, 
               no_repeat_ngram_size=0, 
               num_beams=1, penalty_alpha=0, 
               length_penalty=1, 
               early_stopping=False, 
               seed=0)

