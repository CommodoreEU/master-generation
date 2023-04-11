import re
import sys
from pathlib import Path

import accelerate
import torch

import modules.shared as shared

sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import llama
import llama_inference_offload
import opt

def load_quantized(model_name):
    model_type = 'llama'
    wbits = 4
    groupsize = 128
    
    load_quant = llama.load_quant

    # Now we are going to try to locate the quantized model file.
    path_to_model = Path(f'models/{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]
    else:
        if path_to_model.name.lower().startswith('llama-7b'):
            pt_model = f'llama-7b-{wbits}bit'
        elif path_to_model.name.lower().startswith('llama-13b'):
            pt_model = f'llama-13b-{wbits}bit'
        elif path_to_model.name.lower().startswith('llama-30b'):
            pt_model = f'llama-30b-{wbits}bit'
        elif path_to_model.name.lower().startswith('llama-65b'):
            pt_model = f'llama-65b-{wbits}bit'
        else:
            pt_model = f'{model_name}-{wbits}bit'

        # Try to find the .safetensors or .pt both in models/ and in the subfolder
        for path in [Path(p+ext) for ext in ['.safetensors', '.pt'] for p in [f"models/{pt_model}", f"{path_to_model}/{pt_model}"]]:
            if path.exists():
                print(f"Found {path}")
                pt_path = path
                break

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

   
    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize)

    model = model.to(torch.device('cuda:0'))

    return model
