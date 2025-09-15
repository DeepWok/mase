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

q_config = {
    "by": "type",
    "conv2d": {
        "config": {
            "tile_type": "pcm",
            "core_size": 256,
            "num_bits": 8,
            "programming_noise": True,
            "read_noise": True,
            "ir_drop": True,
            "out_noise": True,
        }
    },
    "linear": {
        "config": {
            "tile_type": "pcm",
            "core_size": 256,
            "num_bits": 8,
            "programming_noise": True,
            "read_noise": False,
            "ir_drop": False,
            "out_noise": False,
        }
    },
}


def test_cim_transform_module_roberta():
    pretrained = "JeremiahZ/roberta-base-mnli"
    model = RobertaForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)

    qmodel, _ = cim_matmul_transform_pass(model, q_config)
    print(model)
    print(qmodel)


def test_cim_transform_module_with_lora_roberta():
    pretrained = "JeremiahZ/roberta-base-mnli"
    model = RobertaForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)

    # lora hyperparameters from https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#hyperparameters-and-recommendations

    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,  # dense first
        "adapter_name": "default",
        "disable_adapter": False,
    }
    qmodel, _ = cim_matmul_transform_pass(model, q_config, lora_config)

    print(model)
    print(qmodel)


if __name__ == "__main__":
    test_cim_transform_module_with_lora_roberta()
