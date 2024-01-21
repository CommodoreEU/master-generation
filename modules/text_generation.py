import gc
import random
import re
import time
import traceback
import ast

import numpy as np
import torch
import transformers

import modules.shared as shared
from modules.callbacks import (Iteratorize, Stream,
                               _SentinelTokenStoppingCriteria)


import matplotlib.pyplot as plt
import hashlib

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def secure_hash_to_numbers(input_string, range_list):

    
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(input_string)

    core = [token.lemma_ for token in doc if not token.is_stop]

    core_str = " ".join(core)
    
    hashed_bytes = hashlib.sha256(core_str.encode()).digest()

    num_numbers = len(range_list)
    hashed_integers = [int.from_bytes(hashed_bytes[i:i+4], byteorder='big') for i in range(0, num_numbers * 4, 4)]

    #cast hash to integer, then use modulo to map to required range
    result_numbers = []
    for i in range(num_numbers):
        range_min, range_max = range_list[i]
        integer = hashed_integers[i]
        mapped_number = (integer % (range_max - range_min + 1)) + range_min
        result_numbers.append(mapped_number)
    
    
    
    
    '''
    import subprocess
    def run_classifier(param):
        # Replace 'your_conda_env' with the name of your Conda environment
        run_classifier_script = f"mamba run -n classifier python classifier.py '{param}'"
        command = f"/bin/bash -c '{run_classifier_script}'"

        # Run the command and capture output
        completed_process = subprocess.run(command, shell=True, text=True, capture_output=True)
        return completed_process.stdout.strip()

    # Rest of your LLM code
    # ...
    # Call the function where needed and pass the parameter
    classifier_output = run_classifier("This is a test.")
    print("Output from classifier:", classifier_output)
    '''
    
    
    '''
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=-1)
    
    labels = [
        "Statement",      # Declarative statements or assertions
        "Question",       # Any form of inquiry or query
        "Request",        # Sentences that ask for something
        "Command",        # Imperative or directive statements
        "Offer",          # Proposals or suggestions
        "Explanation",    # Providing clarifications or reasons
        "Description",    # Descriptive details about something
        "Opinion",        # Personal views or judgments
        "Fact",           # Factual or objective information
        "Assumption",     # Sentences based on assumptions
        "Belief",         # Expressions of belief or conviction
        "Doubt",          # Expressions of uncertainty or skepticism
        "Hope",           # Sentences expressing hope or aspiration
        "Wish",           # Expressions of desires or wishes
        "Fear",           # Statements expressing fear or concern
        "Joy",            # Expressions of happiness or joy
        "Sadness",        # Expressions of sorrow or sadness
        "Anger",          # Statements expressing anger or frustration
        "Surprise",       # Expressions of surprise or astonishment
        "Sarcasm",        # Sarcastic or ironic statements
        "Joke",           # Humorous or joking statements
        "Quote",          # Quotations or cited speech
        "Agreement",      # Expressions of agreement or affirmation
        "Disagreement",   # Expressions of disagreement or dissent
        "Gratitude",      # Expressions of thanks or appreciation
        "Apology"         # Statements of apology or regret
    ]

    labels2 = [
        "Health",
        "Technology",
        "Politics",
        "Economy",
        "Education",
        "Environment",
        "Sports",
        "Travel",
        "Food",
        "Music",
        "Business"
    ]
        

    results = classifier(input_string, labels)
    predicted_label = results['labels'][0]
    label_index = labels.index(predicted_label)

    results2 = classifier(input_string, labels2)
    predicted_label2 = results2['labels'][0]
    label_index2 = labels2.index(predicted_label2)
   
    #print(input_string)
    #print([label_index2, label_index])
    #print("--------------------------")
    #return [label_index2, label_index]
    '''
    
    return result_numbers

def get_last_sentence(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    if sentences:
        return sentences[-1]
    else:
        return None  # Return None if there are no sentences
    
def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found

def get_reply_from_output_ids(output_ids, input_ids, original_question, state, is_chat=False):
    if shared.model_type == 'HF_seq2seq':
        reply = decode(output_ids, state['skip_special_tokens'])
    else:
        new_tokens = len(output_ids) - len(input_ids[0])
        reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])

        # Prevent LlamaTokenizer from skipping a space
        if type(shared.tokenizer) is transformers.LlamaTokenizer and len(output_ids) > 0:
            if shared.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
                reply = ' ' + reply

    #if not is_chat:
    #    reply = apply_extensions('output', reply)

    return reply

def get_max_prompt_length(state):
    max_length = state['truncation_length'] - state['max_new_tokens']
    return max_length

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    
    input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

    # This is a hack for making replies more creative.
    if not add_bos_token and input_ids[0][0] == shared.tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]

    # Llama adds this extra token when the first character is '\n', and this
    # compromises the stopping criteria, so we just remove it
    if type(shared.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
        input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    return input_ids.cuda()


def decode(output_ids, skip_special_tokens=False):
    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)




def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def calc_greenlist_mask(scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

def bias_greenlist_logits(scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias

        #scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias[greenlist_mask]
        return scores


def get_greenlist_ids(input_ids: torch.LongTensor) -> list[float]:


        vocab_permutation = list(range(len(shared.vocab)))
        greenlist_size = 0

        vocab_permutation2 = [0] * len(shared.vocab) #[(n, 0) for n in range(len(shared.vocab))]
        #vocab_dict = {value: 0 for value in shared.vocab} 

        if(shared.new_sentence == True):
            print("//////////// New Sentence, generate acrosticon NOW")
            shared.new_sentence = False

            #ASCII values 65 to 90 represent uppercase letters (A to Z), and values 97 to 122 represent lowercase letters (a to z)
            alphabetic_characters = [chr(i) for i in range(65, 91)] #+ [chr(i) for i in range(97, 123)]

            print(f'''//////////// {alphabetic_characters[shared.secret_key[1]]}''')
            i = 0
            for word in shared.vocab:
                if alphabetic_characters[shared.secret_key[1]] in shared.vocab_decode[i]:

                    vocab_permutation[greenlist_size] = word

                    vocab_permutation2[i] = shared.delta_first
                    greenlist_size += 1
                i += 1
            print(f'''/////////// found words: {greenlist_size}''')
        else:
            i = 0
            for word in shared.vocab:
                if shared.vocab_decode[i].upper() in shared.sensorimotor:
                    
                    if (shared.sensorimotor[shared.vocab_decode[i].upper()][shared.classes[shared.secret_key[0]]] > 2.0):
                        vocab_permutation[greenlist_size] = word

                        vocab_permutation2[i] = shared.delta_char
                        greenlist_size += 1

                    
                i += 1

            
        greenlist_ids = vocab_permutation[:greenlist_size] # new

        return vocab_permutation2
        #return greenlist_ids 
        

def boost_tokens_with_a(input_ids, scores, **kwargs):
   
    batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

    for b_idx in range(input_ids.shape[0]):
            #greenlist_ids = get_greenlist_ids(input_ids[b_idx])

            greenlist_ids = get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids



    permutation_tensor = torch.as_tensor(batched_greenlist_ids)
    permutation_tensor = permutation_tensor.to("cuda:0")
    
    scores[:] = scores[:] + permutation_tensor[:] 
    return scores

#max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, 
def generate_reply(question, state, eos_token=None, stopping_strings=None):
    clear_torch_cache()

    seed = set_manual_seed(state['seed'])
    generate_params = {}
    

    original_question = question
    #print(f"\n\n{question}\n--------------------")



    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']:
        generate_params[k] = state[k]

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    original_input_ids = input_ids
    output = input_ids[0]

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    # Add the encoded tokens to generate_params
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})


    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    stopping_criteria_list = transformers.StoppingCriteriaList()
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
            break


    ##logits code
    generate_params.update({"logits_processor": transformers.LogitsProcessorList([boost_tokens_with_a])})
    shared.vocab=list(shared.tokenizer.get_vocab().values())
    shared.vocab_decode = []
    for word in shared.vocab:
         shared.vocab_decode.append(shared.tokenizer.decode(word,state['skip_special_tokens']))


    # Update generate_params with the eos token and the stopping strings
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = stopping_criteria_list

    reply = ""
    t0 = time.time()
    try:
        stream = True
        if not stream:
            # Generate the entire reply at once.
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                output = output.cuda()

            reply = get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=True)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:
            def generate_with_callback(callback=None, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()

                
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:

                    reply = get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=True)

                    #detect if sentece ended to start new hash and acrostic for next word
                    if((decode(output[-1],state['skip_special_tokens']) == ("." or "!"or "?"))):
                        shared.new_sentence = True

                        last_sentence = get_last_sentence(reply)
                        

                        print("------------------found end of sentence, last sentence is:")
                        print(last_sentence)
    
                        shared.secret_key = secure_hash_to_numbers(last_sentence,[(0, 10), (0, 25)])

                    
                    if output[-1] in eos_token_ids:
                        break

    except Exception:
        traceback.print_exc()

    finally:  
        #new_tokens = len(output) - len(input_ids[0])
        #reply = decode(output[-new_tokens:], state['skip_special_tokens'])
        #print(reply)

        # Find the stopping strings
        all_stop_strings = []
        for st in (stopping_strings, state['custom_stopping_strings']):
            if type(st) is str:
                st = ast.literal_eval(f"[{st}]")

            if type(st) is list and len(st) > 0:
                all_stop_strings += st

        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)

        shared.acrostic = 0
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - original_tokens
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return reply

