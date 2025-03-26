#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys
import dill
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from chop.tools import get_tokenized_dataset, get_trainer
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms import attention_transform_pass, fc_transform_pass
from pathlib import Path
import time

# --------------------------------------------------
#   Model specifications
# --------------------------------------------------

checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
tokenizer.pad_token = tokenizer.eos_token


# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
with open(
    f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb"
) as f:
    model = dill.load(f)


def test_fc_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "lora_fc",
            }
        },
    }
    model = fc_transform_pass(model, "transformer.h.11.attn", pass_args)
    return model


model = test_fc_transform_pass(model)

print(model)
