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

from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass
from pathlib import Path
import time

def measure_inference_speed(model, tokenizer, sample_text, device='cuda', num_warmup=5, num_runs=20):
    """
    Measures average inference time (seconds) for `num_runs` forward passes on a given sample_text.
    """
    model.to(device)
    model.eval()

    # Prepare the inputs
    inputs = tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Warm-up passes
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)

    # Synchronize and start timing
    torch.cuda.synchronize()
    start_time = time.time()

    # Actual timed runs
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)

    # Sync and compute elapsed time
    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / num_runs

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

sample_text = "This is a test input to check inference correctness."

with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
    model = dill.load(f)

# measure_inference_speed(model, tokenizer, sample_text, device='cuda:0')

def test_mla_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mgqa",
                # "kv_heads": 2,
            }
        },
    }
    model, _ = attention_transform_pass(model, pass_args)
    return model

model = test_mla_transform_pass(model)
# measure_inference_speed(model, tokenizer, sample_text, device='cuda:0')
print(model)
trainer = get_trainer(
    model=model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=2,
)
eval_results = trainer.evaluate()
print(f"Evaluation accuracy before fintuning: {eval_results['eval_accuracy']}")



# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")


# mg = MaseGraph(
#     network,
# )
# for param in mg.model.bert.embeddings.parameters():
#     param.requires_grad = False
# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
# mg.export(f"{Path.home()}/Projects/mase/mase_output/bert-mla")

# # Explicit inference test with try-except to capture exact error
# sample_text = "This is a test input to check inference correctness."
# inputs = tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True)
# outputs = network(**inputs)
# print(outputs)
# print(network)



# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# model.config.problem_type = "single_label_classification"
# model.config.pad_token_id = tokenizer.eos_token_id
# # print(model)

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
# with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "wb") as f:
#     dill.dump(model, f)

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

# trainer = get_trainer(
#     model=mla_net,
#     tokenized_dataset=dataset,
#     tokenizer=tokenizer,
#     evaluate_metric="accuracy",
#     num_train_epochs=2,
#     # bf16=True,              # Enable bfloat16 for GPUs that support it
#     # bf16_full_eval=True,    # If you also want to evaluate in bf16
# )
# eval_results = trainer.evaluate()
# print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
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