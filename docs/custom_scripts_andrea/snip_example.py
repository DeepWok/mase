import os
import sys

# Add the src directory to the Python path so that the chop module can be found
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Check for required packages and install them if needed
required_packages = ['dill', 'toml', 'torch', 'torchvision', 'transformers', 'datasets', 'pyctcdecode']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing required packages: {', '.join(missing_packages)}")
    print("Please install the missing packages with:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\nAlternatively, you can run this script with PYTHONPATH set to include the src directory:")
    print(f"PYTHONPATH={src_dir} python {__file__}")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import types
import importlib
from torch.utils.data import DataLoader

import logging
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

from pathlib import Path
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.pruning.prune import prune_transform_pass
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass
from chop.tools import get_tokenized_dataset, get_trainer
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.tools import get_trainer
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)

# Print available pruning methods
import chop.passes.graph.transforms.pruning.prune as prune_mod
print("Available pruning methods:", list(prune_mod.weight_criteria_map["local"]["elementwise"].keys()))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # 1. Load Model and Dataset
    # -------------------------------
    print("Loading model and dataset...")
    checkpoint = "facebook/wav2vec2-base-960h"
    tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "nyalpatel/condensed_librispeech_asr"
    
    # Get tokenized dataset and processor
    tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=checkpoint,
        tokenizer_checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
        return_processor=True,
    )
    
    # Create data collator for batching
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Create CTC decoder from vocabulary
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # Load model
    model = AutoModelForCTC.from_pretrained(checkpoint)
    encoder = model.wav2vec2  # static, FX-friendly
    ctc_head = model.lm_head   # dynamic CTC head
    
    # -------------------------------
    # 2. Create MASE Graph
    # -------------------------------
    print("Creating MASE graph...")
    mg = MaseGraph(
        encoder,
        hf_input_names=[
            "input_values",
            "attention_mask",
        ],
    )
    
    # Initialize metadata
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Add common metadata with dummy input
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # Store the dummy input for SNIP to use later
    mg.common_dummy_in = dummy_in
    
    # Create combined model for training and evaluation
    combined_model = CombinedWav2Vec2CTC(
        encoder=mg.model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # -------------------------------
    # 3. Apply Pruning Pass
    # -------------------------------
    print("Applying pruning...")
    
    # Create pruning configuration
    # You can customize this configuration based on your needs:
    # - method: 'random', 'l1-norm', 'snip', 'movement', 'hwpq'
    # - scope: 'local' or 'global'
    # - sparsity: fraction of weights to prune (0.0 to 1.0)
    # - granularity: 'elementwise'
    #
    # Note: Method-specific preprocessing (like SNIP) is automatically 
    # handled by the backend when needed.
    pruning_config = {
        "weight": {
            "sparsity": 0.5,                # Adjust target sparsity as desired (0.0 to 1.0)
            "method": "snip",               # Choose pruning method: 'random', 'l1-norm', 'snip', etc.
            "scope": "local",               # 'local' or 'global'
            "granularity": "elementwise",   # Currently supports 'elementwise'
        },
        "activation": {
            "sparsity": 0.0,                # Set to 0.0 to disable activation pruning
            "method": "l1-norm",            # Method for activation pruning (if enabled)
            "scope": "local",
            "granularity": "elementwise",
        },
    }
    
    # Apply pruning transform
    # This will automatically handle method-specific preprocessing
    mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
    
    print("\nPruning complete!")
    
    # -------------------------------
    # 4. Evaluate Pruned Model
    # -------------------------------
    print("\nEvaluating pruned model...")
    
    # Prepare evaluation dataset
    if 'test' in tokenized_dataset:
        eval_dataset = tokenized_dataset['test']
    elif 'validation' in tokenized_dataset:
        eval_dataset = tokenized_dataset['validation']
    else:
        eval_dataset = tokenized_dataset['train'].select(range(min(100, len(tokenized_dataset['train']))))
        print("No test/validation set found. Using a subset of training data for evaluation.")
    
    # Create trainer for evaluation
    trainer = get_trainer(
        model=combined_model,
        train_dataset=None,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=combined_model.compute_metrics,
        max_steps=0,  # We're only evaluating, not training
    )
    
    # Run evaluation
    eval_results = trainer.evaluate()
    
    # Print evaluation results
    print("\n-------- Evaluation Results --------")
    print(f"Word Error Rate (WER): {eval_results['eval_wer']:.4f}")
    print(f"Character Error Rate (CER): {eval_results.get('eval_cer', 'N/A')}")
    if 'eval_loss' in eval_results:
        print(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")
    
    # Report parameter counts
    total_params = sum(p.numel() for p in mg.model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in mg.model.parameters())
    sparsity_achieved = 1.0 - (non_zero_params / total_params)
    
    print("\n-------- Sparsity Report --------")
    print(f"Target Sparsity: {pruning_config['weight']['sparsity']:.2%}")
    print(f"Achieved Sparsity: {sparsity_achieved:.2%}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Non-zero Parameters: {non_zero_params:,}")
    print(f"Zero Parameters: {total_params - non_zero_params:,}")


if __name__ == "__main__":
    main() 