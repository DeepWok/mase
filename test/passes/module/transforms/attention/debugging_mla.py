#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import evaluate
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from chop.tools import get_trainer
from chop.passes.module.transforms import attention_transform_pass

#####################################
# 1) CE & PPL evaluation helper
#####################################
def evaluate_ce_and_ppl(trainer, eval_dataset):
    predictions, labels, _ = trainer.predict(eval_dataset)
    logits_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    loss_fn = nn.CrossEntropyLoss()
    cross_entropy_val = loss_fn(logits_tensor, labels_tensor)
    ppl_val = torch.exp(cross_entropy_val)

    return cross_entropy_val.item(), ppl_val.item()

#####################################
# 2) Load dataset + tokenizer
#####################################
dataset_name = "imdb"
raw_dataset = load_dataset(dataset_name)
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # For GPT-2

def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )

tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

#####################################
# 3) Load or train GPT-2
#####################################
model_path = "./gpt2_finetuned_imdb"
if os.path.exists(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

#####################################
# 4) Evaluate GPT-2 CE & PPL
#####################################
trainer_original = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./tmp_original", per_device_eval_batch_size=8, num_train_epochs = 1),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
ce_before, ppl_before = evaluate_ce_and_ppl(trainer_original, tokenized_dataset["test"])
print(f"[BEFORE TRANSFORMATION] CE={ce_before:.4f}, PPL={ppl_before:.4f}")

#####################################
# 5) Transform to MLA
#####################################
def transform_to_mla(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mla",
            }
        },
    }
    mla_model, _ = attention_transform_pass(model, pass_args)
    return mla_model

mla_model = transform_to_mla(model)

#####################################
# 6) Evaluate MLA CE & PPL
#####################################
# Create a new trainer for the MLA model
trainer_mla = Trainer(
    model=mla_model,
    args=TrainingArguments(output_dir="./tmp_mla", per_device_eval_batch_size=8, num_train_epochs = 1),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    # num_train_epochs=5,
)

ce_after, ppl_after = evaluate_ce_and_ppl(trainer_mla, tokenized_dataset["test"])
print(f"[AFTER TRANSFORMATION, NO FINE-TUNING] CE={ce_after:.4f}, PPL={ppl_after:.4f}")

#####################################
# 7) (Optional) Fine-tune MLA Model
#####################################
trainer_mla.train()
ce_ft, ppl_ft = evaluate_ce_and_ppl(trainer_mla, tokenized_dataset["test"])
print(f"[AFTER MLA FINE-TUNING] CE={ce_ft:.4f}, PPL={ppl_ft:.4f}")