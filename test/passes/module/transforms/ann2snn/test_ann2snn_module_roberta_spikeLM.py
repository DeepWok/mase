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

logger = logging.getLogger(__name__)
from chop.passes.module.transforms import quantize_module_transform_pass


import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig

roberta_base_config = {
    "architectures": ["RobertaForMaskedLM"],
    "attention_probs_dropout_prob": 0.1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 514,
    "model_type": "roberta",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "type_vocab_size": 1,
    "vocab_size": 50265,
}

mode_config_path = (
    "/home/thw20/projects/mase/configs/hf_model_configs/configs/roberta_base.json"
)

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
config = AutoConfig.from_pretrained(mode_config_path, cache_dir="/data/models")
logger.info("Training new model from scratch")
model = AutoModelForMaskedLM.from_config(config)
f = open(f"roberta_base_arch.txt", "w")
f.write(str(model))
f.close()

convert_pass_args = {
    "by": "regex_name",
    "roberta\.encoder\.layer\.\d+\.attention\.self": {
        "config": {
            "name": "spikeLM",
            "input_bits": 2,
            "T": 4,
        },
    },
}

mg, _ = ann2snn_module_transform_pass(model, convert_pass_args)


convert_pass_args = {
    "by": "type",
    "linear": {
        "config": {
            "name": "elastic_bi_spiking",
            "input_bits": 2,
            "T": 4,
        },
    },
}

mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)

f = open(f"roberta_base_arch_snn.txt", "w")
f.write(str(mg))
f.close()
