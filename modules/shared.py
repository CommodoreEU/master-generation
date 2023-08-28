import argparse


model = None
tokenizer = None
model_name = "None"
delta = None
vocab = None
vocab_decode = None

groupsize = 128
wbits = 4
model_dir = 'models'
model_type = 'llama'
sensorimotor = None

trust_remote_code = True
code = ""
acrostic = 0
new_sentence = False
delta = 0.0
delta_char = 0.0
delta_first = 0.0
flag = 0