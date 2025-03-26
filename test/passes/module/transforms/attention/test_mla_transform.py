#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from chop.tools import get_tokenized_dataset, get_trainer
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms import (
    quantize_module_transform_pass,
    attention_transform_pass,
)
from pathlib import Path


checkpoint = "openai-community/gpt2"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def test_mla_transform_pass():
    pass_args = {
        "by": "type",
        "gpt2block": {
            "config": {
                "name": "mla",
            }
        },
    }
    mla_network, _ = attention_transform_pass(model, pass_args)
    return mla_network


mla_network = test_mla_transform_pass()
print(mla_network)
