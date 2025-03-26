import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

from pyctcdecode import build_ctcdecoder
from pathlib import Path
from chop.tools import get_tokenized_dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from chop.models import DataCollatorCTCWithPadding
from chop.dataset import MaseDataModule
from datasets import DatasetDict  # Needed to build a DatasetDict
import numpy as np

# Use the new evaluate library to load the WER metric
import evaluate
wer_metric = evaluate.load("wer")

# -------------------------------
# 1. Model & dataset setup
# -------------------------------
checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

# Retrieve tokenized dataset, tokenizer, and processor
tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    tokenizer_checkpoint=checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

# (Optional) To use a small subset for faster training, uncomment the following:
# train_subset = tokenized_dataset["train"].select(range(100))
# test_subset  = tokenized_dataset["test"].select(range(20))
# tokenized_dataset = DatasetDict({"train": train_subset, "test": test_subset})

# Build the CTC decoder from the tokenizer vocabulary
vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

# Load the full model
model = AutoModelForCTC.from_pretrained(checkpoint)

# -------------------------------
# Freeze all parameters, then unfreeze only the allowed layers in the encoder:
# nn.Linear, nn.Conv1d, and nn.Conv2d.
# -------------------------------
for param in model.parameters():
    param.requires_grad = False

for module in model.wav2vec2.modules():
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        for param in module.parameters():
            param.requires_grad = True

# Optionally, log the names of trainable parameters
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
logger.info("Trainable parameters:")
for name in trainable_params:
    logger.info(name)

# -------------------------------
# 2. Data module & collator
# -------------------------------
batch_size = 4
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=checkpoint,
    num_workers=0,
    processor=processor,
)
data_module.setup()

data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True
)

# -------------------------------
# 3. Define compute_metrics for WER
# -------------------------------
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids

    # Replace -100 with pad_token_id so they don't affect decoding
    labels[labels == -100] = tokenizer.pad_token_id

    # Get the argmax over time dimension
    pred_ids = np.argmax(logits, axis=-1)

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer_val = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_val}

# -------------------------------
# 4. Trainer with evaluation each epoch
# -------------------------------
training_args = TrainingArguments(
    output_dir="wav2vec2_checkpoints/",
    num_train_epochs=15,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",    # Evaluate at end of each epoch
    logging_strategy="epoch",
    save_strategy="no",
    report_to=[],                   # Disable wandb, etc.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------------
# 5. Train and print WER for each epoch
# -------------------------------
logger.info("Starting training...")
train_result = trainer.train()
logger.info("Training complete.")

# Print out the WER logged at each epoch
for log_item in trainer.state.log_history:
    if "eval_wer" in log_item:
        epoch = log_item.get("epoch", "?")
        wer_val = log_item["eval_wer"]
        logger.info(f"Epoch {epoch}: WER = {wer_val:.4f}")

final_eval = trainer.evaluate()
logger.info(f"Final evaluation: {final_eval}")
