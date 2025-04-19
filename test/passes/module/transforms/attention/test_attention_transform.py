#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import sys
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms import (
    attention_swap_transform_pass,
)
from pathlib import Path
import time


checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"


# testing disabled due to large model size download, these tests were executed offline

# def test_gqa_transform_pass():

#     model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#     pass_args = {
#         "by": "type",
#         "gpt2spda": {
#             "config": {
#                 "name": "mgqa",
#                 "kv_heads": 2,
#             }
#         },
#     }
#     model, _ = attention_swap_transform_pass(model, pass_args)


# def test_fc_transform_pass():

#     model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#     pass_args = {
#         "by": "name",
#         "transformer.h.11.attn": {
#             "config": {
#                 "name": "lora_fc",
#             }
#         },
#     }
#     model, _ = attention_swap_transform_pass(model, pass_args)


# def test_mla_transform_pass():

#     model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#     pass_args = {
#         "by": "type",
#         "gpt2block": {
#             "config": {
#                 "name": "mla",
#             }
#         },
#     }
#     mla_network, _ = attention_swap_transform_pass(model, pass_args)


# test_gqa_transform_pass()
# test_fc_transform_pass()
# test_mla_transform_pass()
