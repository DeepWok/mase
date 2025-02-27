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
model_name = "google-bert/bert-base-uncased"  # or "prajjwal1/bert-tiny"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)


def test_mla_transform_pass():
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "latent",
            }
        },
    }
    mla_transform_pass(model, pass_args)


test_mla_transform_pass()
