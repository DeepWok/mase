#!/usr/bin/env python3
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
from chop.tools import get_trainer
from chop.passes.module.transforms import attention_transform_pass

checkpoint = "openai-community/gpt2"
dataset_name = "imdb"

# 1) Load raw dataset
raw_dataset = load_dataset(dataset_name)

# 2) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# 3) First, do a broader tokenize to e.g. 256 or 512 so we see the entire text
def initial_tokenize_fn(examples):
    return tokenizer(
        examples["text"] if "text" in examples else examples["sentence"],
        truncation=True,
        padding=False,  # We'll do final padding after slicing
        max_length=512, # just an upper bound
    )

tokenized_dataset = raw_dataset.map(initial_tokenize_fn, batched=True)

# 4) Slice to last 12 tokens
def keep_last_12_tokens(examples):
    # We'll gather the last 12 tokens from "input_ids" and "attention_mask"
    new_input_ids = []
    new_attn_masks = []
    for input_ids, attn_mask in zip(examples["input_ids"], examples["attention_mask"]):
        # keep the last 12 if length > 12
        input_ids = input_ids[-12:]
        attn_mask = attn_mask[-12:]
        new_input_ids.append(input_ids)
        new_attn_masks.append(attn_mask)

    examples["input_ids"] = new_input_ids
    examples["attention_mask"] = new_attn_masks
    return examples

tokenized_dataset = tokenized_dataset.map(keep_last_12_tokens, batched=True)

# 5) Now we do final padding to ensure shape [batch_size, 12]
def pad_to_12(examples):
    return tokenizer.pad(examples, padding="max_length", max_length=12)

tokenized_dataset = tokenized_dataset.map(pad_to_12, batched=True)

# 6) Remove unwanted columns, keep "label"
tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset["train"].column_names if col not in ("input_ids","attention_mask","label")]
)
tokenized_dataset.set_format("torch")

# 7) Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"
model.config.pad_token_id = tokenizer.eos_token_id

def test_mla_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mla",
            }
        },
    }
    mla_network, _ = attention_transform_pass(model, pass_args)
    print("[INFO] Transformed GPT2 model with MLA:")
    print(mla_network)
    return mla_network

mla_net = test_mla_transform_pass(model)

# 8) Trainer
trainer = get_trainer(
    model=mla_net,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=2,
)

eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")
