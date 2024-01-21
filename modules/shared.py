import argparse

classifier = None
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
stop_everything = False

trust_remote_code = True
no_use_fast = False
code = ""
acrostic = 0
new_sentence = False
delta = 0.0
delta_char = 0.0
delta_first = 0.0
flag = 0
secret_key = []
classes = []