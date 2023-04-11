import gc
import re
import time
import traceback

import numpy as np
import torch
import transformers

import modules.shared as shared
from modules.callbacks import (Iteratorize, Stream,
                               _SentinelTokenStoppingCriteria)


import matplotlib.pyplot as plt

def get_max_prompt_length(tokens):
    max_length = 2048-tokens
    return max_length

def encode(prompt, tokens_to_generate=0, add_special_tokens=True):
    input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=get_max_prompt_length(tokens_to_generate), add_special_tokens=add_special_tokens)
    return input_ids.cuda()

def decode(output_ids):
    reply = shared.tokenizer.decode(output_ids, skip_special_tokens=True)
    reply = reply.replace(r'<|endoftext|>', '')
    return reply


def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_manual_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def calc_greenlist_mask(scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

def bias_greenlist_logits(scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

def get_greenlist_ids(input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        #self._seed_rng(input_ids)
        #t0 = time.time()
        
        #vocab=list(shared.tokenizer.get_vocab().values())

        #greenlist_size = int(len(vocab) * gamma)
        #vocab_permutation = torch.randperm(len(vocab), device=input_ids.device)
        
        #greenlist_ids = vocab_permutation[:greenlist_size] # new

        

        
        #vocab_permutation = []#torch.randperm(len(vocab), device=input_ids.device)
        vocab_permutation = list(range(len(shared.vocab)))

        greenlist_size = 0
        i = 0
        for word in shared.vocab:
            if "a" in shared.vocab_decode[i]:
                vocab_permutation[greenlist_size] = word
                greenlist_size += 1
            i += 1

        #greenlist_size = 5
        greenlist_ids = vocab_permutation[:greenlist_size] # new

        #t1 = time.time()
        #print(f"\n mask generated in {(t1-t0):.2f} seconds")
        return greenlist_ids

def boost_tokens_with_a(input_ids, scores, **kwargs):
    # Get the tokenizer and list of token ids
    #tokenizer = kwargs["tokenizer"]

    batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

    for b_idx in range(input_ids.shape[0]):
            greenlist_ids = get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

    # Iterate over each token id and boost the score if the token contains "a"
    #for i, token_id in enumerate(input_ids[0]):
    #    token = decode(token_id.item())
    #    if "a" in token:
    #        scores[0][i] += 15.0
    
    #print(decode(input_ids))


    #print(shared.tokenizer.decode(input_ids[0]))
    #for score in scores:
    #if shared.green_tokens_mask is None:
    green_tokens_mask = calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

    scores = bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=shared.delta)


    
    #print("next logits")
    return scores


def generate_reply(question, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, eos_token=None, stopping_strings=[], delta=0.0):
    clear_torch_cache()
    set_manual_seed(seed)


    t0 = time.time()

    original_question = question
    print(f"\n\n{question}\n--------------------")


    input_ids = encode(question, max_new_tokens)
    original_input_ids = input_ids
    output = input_ids[0]

    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    stopping_criteria_list = transformers.StoppingCriteriaList()
    if type(stopping_strings) is list and len(stopping_strings) > 0:
        t = [encode(string, 0, add_special_tokens=False) for string in stopping_strings]
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=t, starting_idx=len(input_ids[0])))

    generate_params = {}
    generate_params.update({
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_ids,
        "stopping_criteria": stopping_criteria_list,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "typical_p": typical_p,
        "repetition_penalty": repetition_penalty,
        "encoder_repetition_penalty": encoder_repetition_penalty,
        "top_k": top_k,
        "min_length": min_length,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "penalty_alpha": penalty_alpha,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
    })
    generate_params.update({"inputs": input_ids})
    
    
    generate_params.update({"logits_processor": transformers.LogitsProcessorList([boost_tokens_with_a])})
    shared.vocab=list(shared.tokenizer.get_vocab().values())
    shared.vocab_decode = []
    for word in shared.vocab:
         shared.vocab_decode.append(shared.tokenizer.decode(word))

    # Generate the entire reply at once.
    with torch.no_grad():
        output = shared.model.generate(**generate_params)[0]
        output = output.cuda()

    new_tokens = len(output) - len(input_ids[0])
    reply = decode(output[-new_tokens:])
    #print(reply)

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            shared.model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)


    #with generate_with_streaming(**generate_params) as generator:
    #    for output in generator:
            #new_tokens = len(output) - len(input_ids[0])
            #print(decode(output[-1]),end="")
            #reply = decode(output[-new_tokens:])

            #if not (shared.args.chat or shared.args.cai_chat):
            #    reply = original_question + apply_extensions(reply, "output")

            #if output[-1] in eos_token_ids:
            #   break
            #print(reply)

        #print(reply)


    t1 = time.time()
    print(f"\n Output generated in {(t1-t0):.2f} seconds ({(len(output)-len(original_input_ids[0]))/(t1-t0):.2f} tokens/s, {len(output)-len(original_input_ids[0])} tokens, context {len(original_input_ids[0])}) \n")
    return reply









def plot_char_freq(text, n):
    text = text.lower()

    # Count the frequency of each alphabetic character in the text
    freq_dict = {}
    for char in text:
        if char.isalpha():
            freq_dict[char] = freq_dict.get(char, 0) + 1

    # Sort the frequency dictionary by value (frequency)
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

    # Limit the number of characters to plot
    max_n_chars = min(n, len(sorted_freq))

    # Extract the keys (alphabetic characters) and values (frequencies) as separate lists
    chars, freqs = zip(*sorted_freq[:max_n_chars])

    # Calculate the total number of alphabetic characters in the text
    total_chars = sum(freq_dict.values())

    # Calculate the relative frequency of each alphabetic character
    rel_freqs = [freq / total_chars for freq in freqs]


    colors = ["red" if char == "a" else "blue" for char in chars]
    # Add the y-axis values inside the bars
    plt.bar(chars, rel_freqs, align='center', color=colors)
    for i, v in enumerate(rel_freqs):
        plt.text(i, v, "{:.2f}".format(v), ha='center', va='bottom')

    # Set the x-axis and y-axis labels
    plt.xlabel("Alphabetic character")
    plt.ylabel("Relative frequency")


    # Show only the top n characters in the plot title
    if max_n_chars < len(sorted_freq):
        plt.title(f"Top {max_n_chars} most frequent characters")
    else:
        plt.title("All alphabetic characters")

    # Show the plot
    plt.show()