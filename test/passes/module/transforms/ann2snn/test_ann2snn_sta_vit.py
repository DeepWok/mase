#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys
from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
import torch
import torch.nn as nn
from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

resolved_path = Path(__file__).resolve().parents[5].as_posix()
print(resolved_path)
from chop.passes.module.transforms import quantize_module_transform_pass
from chop.passes.module.transforms.replacement import replace_module_transform_pass
import torch
from torch import nn
import json
from transformers import ViTImageProcessor, ViTForImageClassification


vit_class = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float16)

# Define the output file
output_file = "vit_model_arch.txt"
with open(output_file, "w") as f:
    for n, m in vit_class.named_modules():
        f.write(f"{n}: {m}\n")

# tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)
# for param in bert.parameters():
#     param.requires_grad = True  # QAT training


convert_pass_args = {
    "by": "type",
    "gelu": {
        "manual_instantiate": True,
        "config": {
            "name": "gelu_sta",
        },
    },
}


mg, _ = replace_module_transform_pass(vit_class, convert_pass_args)

# output_file = "vit_model_arch_2.txt"
# with open(output_file, "w") as f:
#     for n, m in mg.named_modules():
#         f.write(f"{n}: {m}\n")


convert_pass_args = {
    "by": "regex_name",
    "roberta\.encoder\.layer\.\d+\.attention\.self": {
        "config": {
            "name": "sta",
            "level": 32,
            "neuron_type": "ST-BIF",
        },
    },
}
mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)

convert_pass_args = {
    "by": "type",
    "embedding": {
        "config": {
            "name": "zip_tf",
        },
    },
    "linear": {
        "config": {
            "name": "unfold_bias",
            "level": 32,
            "neuron_type": "ST-BIF",
        },
    },
    "conv2d": {
        "config": {
            "name": "zip_tf",
            "level": 32,
            "neuron_type": "ST-BIF",
        },
    },
    "layernorm": {
        "config": {
            "name": "zip_tf",
        },
    },
    "relu": {
        "manual_instantiate": True,
        "config": {
            "name": "identity",
        },
    },
    "lsqinteger": {
        "manual_instantiate": True,
        "config": {
            "name": "st_bif",
            # Default values. These would be replaced by the values from the LSQInteger module, so it has no effect.
            # "q_threshold": 1,
            # "level": 32,
            # "sym": True,
        },
    },
}
mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)

# f = open(f"spiking_model_arch.txt", "w")
# f.write(str(mg))
# f.close()
