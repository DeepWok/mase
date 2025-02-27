#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms import quantize_module_transform_pass
from chop.passes.module.transforms import mla_transform_pass


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
model_name = "prajjwal1/bert-mini"  # or "prajjwal1/bert-tiny"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)


def test_mla_transform_pass():
    mlp = model()
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "weight_width": 8,
                "weight_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
            }
        },
    }
    mla_transform_pass(mlp, pass_args)


test_mla_transform_pass()
