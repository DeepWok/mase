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

from chop.passes.module.transforms import quantize_module_transform_pass, mla_transform_pass
from pathlib import Path

# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
# model_name = "google-bert/bert-base-uncased"  # or "prajjwal1/bert-tiny"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(model)

checkpoint = "google-bert/bert-base-uncased"
tokenizer_checkpoint = "google-bert/bert-base-uncased"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"
# trainer = get_trainer(
#     model=model,
#     tokenized_dataset=dataset,
#     tokenizer=tokenizer,
#     evaluate_metric="accuracy",
#     num_train_epochs=2,
# )
# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
# mg = MaseGraph(
#     model,
# )
# mg, _ = passes.init_metadata_analysis_pass(mg)
# mg, _ = passes.add_common_metadata_analysis_pass(mg)
# mg.export(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch")

def test_mla_transform_pass():
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    pass_args = {
        "by": "type",
        "bert_attention": {
            "config": {
                "name": "latent",
            }
        },
    }
    mla_network, _ = mla_transform_pass(model, pass_args)
    print(mla_network)
    return mla_network

mla_net = test_mla_transform_pass()

trainer = get_trainer(
    model=mla_net,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=2,
    # bf16=True,              # Enable bfloat16 for GPUs that support it
    # bf16_full_eval=True,    # If you also want to evaluate in bf16
)
eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
# mg = MaseGraph(
#     mla_net,
# )
# mg.export(f"{Path.home()}/Projects/mase/mase_output/bert-mla")
# for param in mg.model.bert.embeddings.parameters():
#     param.requires_grad = False
# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")

# mg.export(f"{Path.home()}/tutorial_2_sft")