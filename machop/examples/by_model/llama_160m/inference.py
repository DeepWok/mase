"""
A toy llama with 160M parameters
"""

import os
import sys

import torch

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(os.path.join("..", "..", "..", "..", "machop"))


from chop.models.manual.llama_quantized import (
    LlamaQuantizedConfig,
    LlamaQuantizedForCausalLM,
)
from transformers.models.llama import LlamaTokenizer

name = "Cheng98/llama-160m"
device = "cuda"
config = LlamaQuantizedConfig.from_pretrained(
    name,
    # quant_config="./quant_config_na.toml"
)
llama = LlamaQuantizedForCausalLM.from_pretrained(
    name,
    config=config,
)
tokenizer = LlamaTokenizer.from_pretrained(name)

text = "Hi, my name is"
inputs = tokenizer(
    text,
    return_tensors="pt",
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    max_length=32,
)
inputs = {k: v.to(device) for k, v in inputs.items()}

llama.to(device)
llama.eval()
with torch.no_grad():
    outputs = llama(**inputs)
    _, pred_ids = torch.max(outputs.logits, dim=-1)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

config.save_pretrained("./llama-160m")
print(llama)
print(pred_text)
