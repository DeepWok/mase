#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from chop.nn.attention.modules import LlamaForCausalLM
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes
from chop.passes.module.transforms import (
    quantize_module_transform_pass,
    attention_transform_pass,
)
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())



model_path = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        attn_implementation="sdpa",
        partial_rotary_factor=8  # Using custom parameter
    )


def test_llama_transform_pass(model):
    pass_args = {
        "by": "model",
        "llama": {
            "config": {}
        },
    }
    mla_network, _ = attention_transform_pass(model, pass_args)
    return mla_network

model = test_llama_transform_pass(model)
print(model)