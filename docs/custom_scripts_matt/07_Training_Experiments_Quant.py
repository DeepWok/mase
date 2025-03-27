import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from pathlib import Path
from chop.tools import get_tokenized_dataset  # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from chop import MaseGraph
import chop.passes as passes  # type: ignore
from chop.passes import add_movement_metadata_analysis_pass
from chop.passes.module import report_trainable_parameters_analysis_pass  # type: ignore
from chop.passes.graph.transforms.pruning import MovementTrackingCallback
from chop.tools import get_trainer  # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)

import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

from pyctcdecode import build_ctcdecoder
from transformers import TrainingArguments
import evaluate

# Load the WER metric from evaluate
wer_metric = evaluate.load("wer")

# -------------------------------
# 1. Model & Dataset Setup
# -------------------------------
checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

# Get tokenized dataset, tokenizer, and processor
tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    tokenizer_checkpoint=checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

# Build the model and separate encoder and CTC head
model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2    # FX-friendly encoder
ctc_head = model.lm_head    # Dynamic CTC head
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# -------------------------------
# 2. Data Module, FX Graph & Quantization
# -------------------------------
batch_size = 4

mg = MaseGraph(
    encoder,
    hf_input_names=[
        "input_values",
        "attention_mask",
    ],
)

# Initialize metadata analysis pass
mg, _ = passes.init_metadata_analysis_pass(mg)

dummy_in = {
    "input_values": torch.zeros((1, 16000), dtype=torch.float32),
    "attention_mask": torch.ones((1, 16000), dtype=torch.long),
}

# Add common metadata to the FX graph
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_in,
        "add_value": True,
        "force_device_meta": False,
    }
)

# Define quantization configuration
quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # Data quantization parameters
            "data_in_width": 16,
            "data_in_frac_width": 8,
            # Weight quantization parameters
            "weight_width": 16,
            "weight_frac_width": 8,
            # Bias quantization parameters
            "bias_width": 16,
            "bias_frac_width": 8,
        }
    },
}

# Apply the quantization pass
mg, _ = passes.quantize_transform_pass(
    mg,
    pass_args=quantization_config,
)

# -------------------------------
# 2b. Create Combined Model with Quantized Encoder
# -------------------------------
# Build a decoder using pyctcdecode.
# The vocabulary is built from the tokenizer's vocabulary keys.
vocab_list = list(tokenizer.get_vocab().keys())
decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path=None,  # No external language model provided
    alpha=0,
    beta=0,
)

# Create a combined model using the quantized encoder (mg.model),
# the original CTC head, and the decoder.
combined_model = CombinedWav2Vec2CTC(
    encoder=mg.model,
    ctc_head=ctc_head,
    decoder=decoder,
    beam_width=10
)

# -------------------------------
# 3. Trainer Setup Using get_trainer Wrapper
# -------------------------------
# Using the get_trainer wrapper from Script Two, which includes additional logic
# to handle data collation and evaluation properly.
trainer = get_trainer(
    model=combined_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="wer",
    data_collator=data_collator,
    num_train_epochs=15,                # Number of epochs
    gradient_accumulation_steps=4,      # As in the snippet from Script Two
    per_device_train_batch_size=2,      # Adjusted batch sizes
    per_device_eval_batch_size=2,
)

# -------------------------------
# 4. Train & Evaluate
# -------------------------------
logger.info("Starting training...")
train_result = trainer.train()
logger.info("Training complete.")

# Log the WER after each evaluation epoch (if available in log history)
for log_item in trainer.state.log_history:
    if "eval_wer" in log_item:
        epoch = log_item.get("epoch", "?")
        wer_val = log_item["eval_wer"]
        logger.info(f"Epoch {epoch}: WER = {wer_val:.4f}")

# Final evaluation
final_eval = trainer.evaluate()
logger.info(f"Final evaluation: {final_eval}")
