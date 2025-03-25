import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import types
import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from datetime import datetime
from torch.utils.data import DataLoader

import logging
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

from pathlib import Path
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.pruning.prune import prune_transform_pass
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
from chop.passes.graph.transforms.pruning.prune_movment_helper import MovementTrackingCallback
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
    calculate_avg_bits_mg_analysis_pass,
)

# Print available pruning methods
import chop.passes.graph.transforms.pruning.prune as prune_mod
print("Available pruning methods:", list(prune_mod.weight_criteria_map["local"]["elementwise"].keys()))


def get_achieved_sparsity(model):
    """Calculate the actual achieved sparsity in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    return 1.0 - (non_zero_params / total_params)


def train_model_with_movement_tracking(model, dataloader, num_epochs=1, learning_rate=5e-5):
    """Train the model while tracking weight movement."""
    print("\n== Training model with movement tracking ==")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize movement tracking using the existing callback
    movement_tracker = MovementTrackingCallback()
    # Call on_train_begin manually since we're not using a Trainer
    movement_tracker.on_train_begin(None, None, None, model=model)
    
    # Move model to device
    device = next(model.parameters()).device
    
    # Train for specified epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        # Track progress
        running_loss = 0.0
        num_batches = 0
        
        # Train on batches
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update movement tracking
            movement_tracker.on_step_end(None, None, None, model=model)
            
            # Log progress
            running_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                
            # Stop early for quick demonstration
            if batch_idx >= 49:  # Process 50 batches for quick testing
                break
                
        # Epoch summary
        epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
    
    print("Training with movement tracking completed.")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create CTC decoder from vocabulary
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # Load model
    model = AutoModelForCTC.from_pretrained(checkpoint)
    encoder = model.wav2vec2  # static, FX-friendly
    ctc_head = model.lm_head   # dynamic CTC head
    
    # Create data module for runtime analysis
    batch_size = 2
    data_module = MaseDataModule(
        name=dataset_name,
        batch_size=batch_size,
        model_name=checkpoint,
        num_workers=0,
        processor=processor
    )
    data_module.setup()
    
    # Define methods and sparsity levels to test
    pruning_methods = ["random", "l1-norm", "snip", "movement"]  # Added movement back
    sparsity_levels = [0.3, 0.5, 0.7, 0.9]      # Use fewer levels for faster testing
    
    # Create runtime analysis config
    runtime_analysis_config = {
        "num_batches": 5,
        "num_GPU_warmup_batches": 2,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": batch_size,
        "sample_rate": 16000,
        "ctc_head": ctc_head
    }
    
    # Results storage
    results = {
        "method": [],
        "sparsity_target": [],
        "sparsity_achieved": [],
        "wer": [],
        "inference_time": [],
        "rtf": [],
        "gpu_power": [],
        "energy": []
    }
    
    # Create a timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Create output directory
    output_dir = "/content/pruning_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # -------------------------------
    # 1.5 Train model to get movement data
    # -------------------------------
    print("\n== Preparing model for movement pruning ==")
    # Create a dataloader for training
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorCTCWithPadding(processor=processor, padding=True)
    )
    
    # Train model with movement tracking
    movement_trained_model = train_model_with_movement_tracking(
        model=copy.deepcopy(model),
        dataloader=train_dataloader,
        num_epochs=1  # Just one epoch for demonstration
    )
    
    # Extract encoder for pruning
    movement_encoder = movement_trained_model.wav2vec2
    
    # -------------------------------
    # 2. Run analysis for each method and sparsity
    # -------------------------------
    print("\nStarting pruning method comparison analysis...")
    
    # First run baseline without pruning to get reference metrics
    print("\n== Baseline (no pruning) ==")
    
    # Create a fresh MASE graph
    mg_baseline = MaseGraph(
        copy.deepcopy(encoder),
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize and add metadata
    mg_baseline, _ = passes.init_metadata_analysis_pass(mg_baseline)
    
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg_baseline, _ = passes.add_common_metadata_analysis_pass(
        mg_baseline,
        pass_args={"dummy_in": dummy_in, "add_value": True, "force_device_meta": False}
    )
    mg_baseline.common_dummy_in = dummy_in
    
    # Run runtime analysis on baseline
    print("Running runtime analysis on baseline model...")
    _, baseline_results = runtime_analysis_pass(mg_baseline, pass_args=runtime_analysis_config)
    
    baseline_wer = baseline_results.get("Average WER", 0.0)
    baseline_latency = baseline_results.get("Average Latency", 0.0)
    baseline_rtf = baseline_results.get("Average RTF", 0.0)
    
    print(f"Baseline WER: {baseline_wer:.4f}")
    print(f"Baseline Latency: {baseline_latency:.2f} ms")
    print(f"Baseline RTF: {baseline_rtf:.4f}")
    
    # Now run each pruning method at different sparsity levels
    for method in pruning_methods:
        print(f"\n== Testing pruning method: {method} ==")
        
        for sparsity in sparsity_levels:
            if sparsity == 0.0:  # Skip 0.0 sparsity as we already did baseline
                continue
                
            print(f"\nSparsity level: {sparsity:.1f}")
            
            # Create a fresh MASE graph with appropriate model
            if method == "movement":
                # Use the model with movement data
                mg = MaseGraph(
                    copy.deepcopy(movement_encoder),
                    hf_input_names=["input_values", "attention_mask"],
                )
                print("Using model with movement data")
            else:
                # Use the original model
                mg = MaseGraph(
                    copy.deepcopy(encoder),
                    hf_input_names=["input_values", "attention_mask"],
                )
            
            # Initialize metadata
            mg, _ = passes.init_metadata_analysis_pass(mg)
            
            # Add common metadata with dummy input
            mg, _ = passes.add_common_metadata_analysis_pass(
                mg,
                pass_args={"dummy_in": dummy_in, "add_value": True, "force_device_meta": False}
            )
            
            # Store the dummy input for SNIP to use later
            mg.common_dummy_in = dummy_in
            
            # Create pruning configuration
            pruning_config = {
                "weight": {
                    "sparsity": sparsity,
                    "method": method,
                    "scope": "local",
                    "granularity": "elementwise",
                },
                "activation": {
                    "sparsity": 0.0,  # Focus on weight pruning only
                    "method": "l1-norm",
                    "scope": "local",
                    "granularity": "elementwise",
                },
            }
            
            # Apply pruning
            mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
            print(f"Applied {method} pruning with sparsity {sparsity:.1f}")
            
            # Calculate achieved sparsity
            sparsity_achieved = get_achieved_sparsity(mg.model)
            print(f"Target sparsity: {sparsity:.1f}, Achieved sparsity: {sparsity_achieved:.2f}")
            
            # Run runtime analysis (this includes WER evaluation)
            print("Running runtime analysis...")
            _, runtime_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
            
            # Extract metrics from runtime analysis
            wer = runtime_results.get("Average WER", 0.0)
            latency = runtime_results.get("Average Latency", 0.0)
            rtf = runtime_results.get("Average RTF", 0.0)
            gpu_power = runtime_results.get("Average GPU Power Usage", 0.0)
            energy = runtime_results.get("Inference Energy Consumption", 0.0)
            
            print(f"WER: {wer:.4f} (vs baseline {baseline_wer:.4f})")
            print(f"Latency: {latency:.2f} ms (vs baseline {baseline_latency:.2f} ms)")
            print(f"RTF: {rtf:.4f} (vs baseline {baseline_rtf:.4f})")
            
            # Store results
            results["method"].append(method)
            results["sparsity_target"].append(sparsity)
            results["sparsity_achieved"].append(sparsity_achieved)
            results["wer"].append(wer)
            results["inference_time"].append(latency)
            results["rtf"].append(rtf)
            results["gpu_power"].append(gpu_power)
            results["energy"].append(energy)
            
            # Save intermediate results to prevent data loss
            np.save(f"{output_dir}/pruning_comparison_results_{timestamp}.npy", results)
    
    # Add baseline to results for plotting
    for method in pruning_methods:
        results["method"].append(method)
        results["sparsity_target"].append(0.0)
        results["sparsity_achieved"].append(0.0)
        results["wer"].append(baseline_wer)
        results["inference_time"].append(baseline_latency)
        results["rtf"].append(baseline_rtf)
        results["gpu_power"].append(baseline_results.get("Average GPU Power Usage", 0.0))
        results["energy"].append(baseline_results.get("Inference Energy Consumption", 0.0))
    
    # -------------------------------
    # 3. Plot results
    # -------------------------------
    print("\nGenerating plots...")
    
    # Convert results to numpy arrays for easier manipulation
    methods = np.array(results["method"])
    sparsity_targets = np.array(results["sparsity_target"])
    sparsity_achieved = np.array(results["sparsity_achieved"])
    wer_values = np.array(results["wer"])
    inference_times = np.array(results["inference_time"])
    rtf_values = np.array(results["rtf"])
    
    # Create a figure for each metric
    plt.figure(figsize=(12, 8))
    
    # Plot WER vs Sparsity for each method
    for method in pruning_methods:
        mask = methods == method
        plt.plot(sparsity_targets[mask], wer_values[mask], 'o-', label=method)
    
    plt.title('Word Error Rate vs Sparsity')
    plt.xlabel('Target Sparsity')
    plt.ylabel('Word Error Rate (WER)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/pruning_wer_comparison_{timestamp}.png")
    plt.close()
    
    # Plot Inference Time vs Sparsity
    plt.figure(figsize=(12, 8))
    for method in pruning_methods:
        mask = methods == method
        plt.plot(sparsity_targets[mask], inference_times[mask], 'o-', label=method)
    
    plt.title('Inference Time vs Sparsity')
    plt.xlabel('Target Sparsity')
    plt.ylabel('Inference Time (ms/sample)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/pruning_inference_time_comparison_{timestamp}.png")
    plt.close()
    
    # Plot RTF vs Sparsity
    plt.figure(figsize=(12, 8))
    for method in pruning_methods:
        mask = methods == method
        plt.plot(sparsity_targets[mask], rtf_values[mask], 'o-', label=method)
    
    plt.title('Real-Time Factor vs Sparsity')
    plt.xlabel('Target Sparsity')
    plt.ylabel('Real-Time Factor (RTF)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/pruning_rtf_comparison_{timestamp}.png")
    plt.close()
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Results saved as {output_dir}/pruning_comparison_results_{timestamp}.npy")
    print(f"Plots saved as {output_dir}/pruning_wer_comparison_{timestamp}.png, {output_dir}/pruning_inference_time_comparison_{timestamp}.png, and {output_dir}/pruning_rtf_comparison_{timestamp}.png")


if __name__ == "__main__":
    main() 
