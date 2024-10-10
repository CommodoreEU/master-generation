import time
import os
import json
import spacy
import hashlib
import random
import math
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import norm, binom
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Change the working directory
os.chdir("/home/feline/master-generation")

# Initialize SpaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    raise OSError("SpaCy model 'en_core_web_sm' not found. Please install it using:\n"
                  "python -m spacy download en_core_web_sm")

# Load sensorimotor norms
try:
    df = pd.read_csv('updated_word_frequencies_with_percent.csv', header=0)
except FileNotFoundError:
    raise FileNotFoundError("The file 'updated_word_frequencies_with_percent.csv' was not found.")

# Setup shared sensorimotor data
shared = type('', (), {})()
shared.sensorimotor = df.set_index('Word').T.to_dict('dict')
shared.classes = ['Auditory', 'Gustatory', 'Haptic', 'Interoceptive', 'Olfactory', 'Visual', 'Foot_leg', 'Hand_arm', 'Head', 'Mouth', 'Torso']
shared.secret_key = [0, 0]

# Functions
def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def secure_hash_for_word(word, range_min, range_max):
    hashed_word_bytes = hashlib.sha256(word.encode()).digest()
    hashed_word_int = int.from_bytes(hashed_word_bytes[:4], byteorder='big')
    return (hashed_word_int % (range_max - range_min + 1)) + range_min

# Loading the OPT-2.7B model
print("\nLoading OPT-2.7B model for perplexity calculation...")
try:
    tokenizer_opt = shared.model
    model_opt = shared.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    raise RuntimeError(f"Failed to load OPT-2.7B model: {e}")

# Function to calculate perplexity
def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), reduction='mean')
    return torch.exp(loss).item()

# Detection and perplexity calculation
generated_json_path = 'generation_results.json'
detection_json_path = 'detection_results.json'

try:
    with open(generated_json_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    print(f"Successfully loaded '{generated_json_path}'.")
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{generated_json_path}' was not found.")
except json.JSONDecodeError as e:
    raise ValueError(f"Error decoding JSON from '{generated_json_path}': {e}")

detection_results = []

for idx, sample in enumerate(generated_data, start=1):
    prompt = sample.get('prompt', '')
    print(f"\nProcessing Sample {idx}:")
    sample_detection = {
        'prompt': prompt,
        'outputs': {},
        'detection': {}
    }

    for key, text in sample.items():
        if key == 'prompt':
            continue
        sample_detection['outputs'][key] = text

    detection_results.append(sample_detection)

try:
    with open(detection_json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=4)
    print(f"\nDetection results saved to '{detection_json_path}'.")
except Exception as e:
    print(f"\nFailed to save detection results: {e}")
