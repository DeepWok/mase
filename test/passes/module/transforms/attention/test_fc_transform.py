#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
sys.path.insert(0, src_path)

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

from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass
# FC
from chop.passes.module.transforms import fc_transform_pass

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

# with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
#     model = dill.load(f)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

def test_mla_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mgqa",
            }
        },
    }
    model, _ = attention_transform_pass(model, pass_args)
    return model

def test_fc_transform_pass(model):
    #替换第0层 attention。
    
    module_name = "bert.encoder.layer.0.attention"  # bert
    #module_name = "transformer.h.0.attn" #gpt2
    model = fc_transform_pass(model, module_name, config={})
    return model

model = test_mla_transform_pass(model)
print("Model after MLA Transform:", model)

model = test_fc_transform_pass(model)
print("Model after FC Transform:", model)

trainer = get_trainer(
    model=model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=2,
)
eval_results = trainer.evaluate()
print(f"Evaluation accuracy before fintuning: {eval_results['eval_accuracy']}")

trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
