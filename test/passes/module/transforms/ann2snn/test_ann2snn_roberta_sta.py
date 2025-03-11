#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
from chop.passes.module.transforms.replacement import replace_module_transform_pass
import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


from chop.passes.module.transforms import quantize_module_transform_pass


import torch
from torch import nn
from transformers import RobertaForSequenceClassification, AutoTokenizer
import json

pretrained = "XianYiyk/roberta-relu-pretrained-sst2"
bert = RobertaForSequenceClassification.from_pretrained(pretrained, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)
for param in bert.parameters():
    param.requires_grad = True  # QAT training

# Define the output file
output_file = "roberta_model_arch.txt"
with open(output_file, "w") as f:
    for n, m in bert.named_modules():
        f.write(f"{n}: {m}\n")


convert_pass_args = {
    "by": "type",
    "gelu": {
        "manual_instantiate": True,
        "config": {
            "name": "gelu_sta",
        },
    },
}


mg, _ = replace_module_transform_pass(bert, convert_pass_args)

output_file = "roberta_model_arch_2.txt"
with open(output_file, "w") as f:
    for n, m in mg.named_modules():
        f.write(f"{n}: {m}\n")

convert_pass_args = {
    "by": "type",
    "attention": {
        "config": {
            "name": "sta",
            "T": 32,
            "bipolar_with_memory" : True, 
            "burst_T": 2
        },
    },
    "layernorm" : {
        "config": {
            "name": "sta",
            "T": 32,
            "bipolar_with_memory" : True, 
            "burst_T": 2
        },
    },
    "linear" : {
        "config": {
            "name": "sta",
            "T": 32,
        },
    },

}

mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)

convert_pass_args = {
    "by": "regex_name",
    
    "roberta\.encoder\.layer\.\d+\.intermediate": {
        "manual_instantiate": True,
        "config": {
            "name": "relu_sta",
            "bipolar_with_memory" : True, 
            "burst_T": 2
        }
    },

}

mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)


output_file = "roberta_model_arch_3_test.txt"
with open(output_file, "w") as f:
    for n, m in mg.named_modules():
        f.write(f"{n}: {m}\n")