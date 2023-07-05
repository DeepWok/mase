"""
This script loads Vicuna-7B-v1.3 and performs inference on a sample text.
!: Sufficient GPU/CPU memory is required to load the model.
"""
import os
import sys

import torch

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(os.path.join("..", "..", "..", "..", "machop"))


from chop.models.manual.llama_quantized import (LlamaQuantizedConfig,
                                                LlamaQuantizedForCausalLM)
from transformers import AutoTokenizer

config = LlamaQuantizedConfig.from_pretrained(
    "lmsys/vicuna-7b-v1.3", quant_config="./quant_config_minimal.toml"
)

max_memory_mapping = {
    0: "23GB",
    1: "23GB",
    2: "23GB",
    3: "23GB",
}
vicuna = LlamaQuantizedForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.3",
    config=config,
    device_map="auto",
    max_memory=max_memory_mapping,
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")

text = "Hi there, would you like to play a game?"
inputs = tokenizer(
    text,
    return_tensors="pt",
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    max_length=32,
)
vicuna.eval()
with torch.no_grad():
    outputs = vicuna(**inputs)
    _, pred_ids = torch.max(outputs.logits, dim=-1)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
print(vicuna)
print(pred_text)
