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
from chop.passes.module.transforms.replacement import replace_module_transform_pass

import clip
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
clip_model,preprocess = clip.load('ViT-B/32', device = device)
vit_class = clip_model.visual
classifier = None

# # Define the output file
# output_file = "ViT-B_32_model_arch.txt"
# with open(output_file, "w") as f:
#     for n, m in vit_class.named_modules():
#         f.write(f"{n}: {m}\n")

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

# output_file = "ViT-B_32_model_arch_2.txt"
# with open(output_file, "w") as f:
#     for n, m in mg.named_modules():
#         f.write(f"{n}: {m}\n")


convert_pass_args = {
    "by": "type",
    "attention": {
        "manual_instantiate": True,
        "config": {
            "name": "attn_sta",
            "T": 32,
            "embed_dim" : 768, 
            "num_heads" : 768, 
            "batch_first": True, 
            "bipolar_with_memory" : True, 
            "burst_T": 2
        },
    },
    "layernorm" : {
        "manual_instantiate": True,
        "config": {
            "name": "layernorm_sta",
            "T": 32,
            "normalized_shape": 768, 
            "eps": 1e-05, 
            "elementwise_affine": True,
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
    "by": "bundle",
    "convert 1": {
        "bundle" : ["Not_None", "relu"],
        "manual_instantiate": True,
        "config": {
            "name": "sta",
            "T": 32,
        },
    }
}

mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)


output_file = "ViT-B_32_model_arch_4.txt"
with open(output_file, "w") as f:
    for n, m in mg.named_modules():
        f.write(f"{n}: {m}\n")

# convert_pass_args = {
#     "by": "type",
#     "embedding": {
#         "config": {
#             "name": "zip_tf",
#         },
#     },
#     "linear": {
#         "config": {
#             "name": "unfold_bias",
#             "level": 32,
#             "neuron_type": "ST-BIF",
#         },
#     },
#     "conv2d": {
#         "config": {
#             "name": "zip_tf",
#             "level": 32,
#             "neuron_type": "ST-BIF",
#         },
#     },
#     "layernorm": {
#         "config": {
#             "name": "zip_tf",
#         },
#     },
#     "relu": {
#         "manual_instantiate": True,
#         "config": {
#             "name": "identity",
#         },
#     },
#     "lsqinteger": {
#         "manual_instantiate": True,
#         "config": {
#             "name": "st_bif",
#             # Default values. These would be replaced by the values from the LSQInteger module, so it has no effect.
#             # "q_threshold": 1,
#             # "level": 32,
#             # "sym": True,
#         },
#     },
# }
# mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)

# f = open(f"spiking_model_arch.txt", "w")
# f.write(str(mg))
# f.close()
