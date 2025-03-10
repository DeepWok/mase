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


from chop.passes.module.transforms import quantize_module_transform_pass


import torch
from torch import nn
from transformers import RobertaForSequenceClassification, AutoTokenizer

pretrained = "XianYiyk/roberta-relu-pretrained-sst2"
bert = RobertaForSequenceClassification.from_pretrained(pretrained, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)
for param in bert.parameters():
    param.requires_grad = True  # QAT training


# def test_ann2snn_module_transform_pass():
quan_pass_args = {
    "by": "regex_name",
    "roberta\.encoder\.layer\.\d+\.attention\.self": {
        "config": {
            "name": "lsqinteger",
            "level": 32,
        }
    },
    "roberta\.encoder\.layer\.\d+\.attention\.output": {
        "config": {
            "name": "lsqinteger",
            "level": 32,
        }
    },
    "roberta\.encoder\.layer\.\d+\.output": {
        "config": {
            "name": "lsqinteger",
            "level": 32,
        }
    },
    "roberta\.encoder\.layer\.\d+\.intermediate": {
        "config": {
            "name": "lsqinteger",
            "level": 32,
        }
    },
    "classifier": {
        "config": {
            "name": "lsqinteger",
            "level": 32,
        }
    },
}
mg, _ = quantize_module_transform_pass(bert, quan_pass_args)
# f = open(f"qann_model_arch.txt", "w")
# f.write(str(mg))
# f.close()

convert_pass_args = {
    "by": "regex_name",
    "roberta\.encoder\.layer\.\d+\.attention\.self": {
        "config": {
            "name": "zip_tf",
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
