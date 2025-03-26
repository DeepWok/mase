#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys
import dill
import torch.nn as nn
from pathlib import Path
import time
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from chop.tools import get_tokenized_dataset, get_trainer
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes
from chop.passes.module.transforms import attention_transform_pass, fc_transform_pass

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


checkpoint = "openai-community/gpt2"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def test_fc_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "lora_fc",
            }
        },
    }
    # model = fc_transform_pass(model, "transformer.h.11.attn", pass_args)
    model = attention_transform_pass(model, pass_args)
    return model


model = test_fc_transform_pass(model)
print(model)
