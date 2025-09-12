#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog

import torch
import torch.nn as nn

from pathlib import Path

from chop.passes.module.transforms import cim_matmul_transform_pass

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, AutoTokenizer
import yaml

pretrained = "JeremiahZ/roberta-base-mnli"
model = RobertaForSequenceClassification.from_pretrained(
    pretrained, num_labels=2, ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)

q_config = yaml.load(
    open("/home/cx922/mase/configs/cim/pcm.yaml", "r"), Loader=yaml.FullLoader
)
qmodel, _ = cim_matmul_transform_pass(model, q_config)

print(model)
print(qmodel)
