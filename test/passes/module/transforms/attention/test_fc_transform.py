#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
# sys.path.insert(0, src_path)
import dill
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from chop.tools import get_tokenized_dataset, get_trainer
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass
# FC
from chop.passes.module.transforms.attention import fc_transform_pass

# --------------------------------------------------
#   Model specifications
# --------------------------------------------------

checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
dataset_name = "wikitext"  

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
tokenizer.pad_token = tokenizer.eos_token

# with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
#     model = dill.load(f)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

def test_fc_transform_pass(model):
    module_name = "transformer.h.11.attn"
    config = {}  # hidden_size
    model = fc_transform_pass(model, module_name, config={})
    return model

model = test_fc_transform_pass(model)
print("Model after FC Output Projection Transform:", model)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)
