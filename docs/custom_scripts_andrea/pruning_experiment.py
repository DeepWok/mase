import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import logging
import matplotlib.pyplot as plt
import seaborn as sns
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

from pathlib import Path
from chop.tools import get_tokenized_dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from chop import MaseGraph
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass
from chop.passes.graph.transforms.pruning import MovementTrackingCallback
from chop.tools import get_trainer
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    calculate_avg_bits_mg_analysis_pass,
)
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
from chop.passes.graph.transforms.pruning.prune_movment_helper import MovementTrackingCallback
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. Define the model and dataset
# -------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

# Get tokenized dataset
tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    tokenizer_checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

# Take a smaller subset of the dataset for faster training
train_subset = tokenized_dataset["train"].select(range(200))  # Use only 200 samples
test_subset = tokenized_dataset["test"].select(range(100))     # Use only 100 samples
tokenized_dataset = DatasetDict({
    "train": train_subset,
    "test": test_subset
})

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

# -------------------------------
# 2. Import dataset
# -------------------------------

batch_size = 4  # Increased batch size since we have less data
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=checkpoint,
    num_workers=0,
    processor=processor
)
data_module.setup()

# -------------------------------
# 3. Helper function to compute sparsity
# -------------------------------
def compute_actual_sparsity(model: nn.Module) -> float:
    total_params = 0
    total_zeros = 0
    for param in model.parameters():
        total_params += param.numel()
        total_zeros += (param == 0).sum().item()
    return total_zeros / total_params if total_params > 0 else 0.0

def setup_model_and_graph():
    """Set up the model and MASE graph."""
    model = AutoModelForCTC.from_pretrained(checkpoint)
    model.config.gradient_checkpointing = True
    encoder = model.wav2vec2
    ctc_head = model.lm_head

    mg = MaseGraph(
        encoder,
        hf_input_names=["input_values", "attention_mask"],
    )

    mg, _ = passes.init_metadata_analysis_pass(mg)

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

    return mg, ctc_head

def run_pruning_experiment(
    pruning_methods: List[str] = ["random", "l1-norm", "movement", "snip", "hwpq"],
    sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    num_train_epochs: int = 2,  # Changed to 2 fine tuning epochs
    save_results: bool = True
) -> Dict:
    """Run the pruning comparison experiment."""
    results = {}

    runtime_analysis_config = {
        "num_batches": 15,
        "num_GPU_warmup_batches": 2,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": batch_size,
        "sample_rate": 16000,
    }

    print("\n" + "="*80)
    print("Starting Pruning Comparison Experiment")
    print("="*80)
    print(f"Methods to test: {', '.join(pruning_methods)}")
    print(f"Sparsity levels: {', '.join([str(s) for s in sparsity_levels])}")
    print("="*80 + "\n")

    for method in pruning_methods:
        print(f"\n{'='*40}")
        print(f"Testing {method.upper()} pruning method")
        print(f"{'='*40}")
        method_results = {}

        for sparsity in sparsity_levels:
            print(f"\n--- Testing sparsity level: {sparsity:.2f} ---")
            
            # Reset model for each experiment
            mg, ctc_head = setup_model_and_graph()
            runtime_analysis_config["ctc_head"] = ctc_head
            
            # Create combined model
            combined_model = CombinedWav2Vec2CTC(
                encoder=mg.model,
                ctc_head=ctc_head,
                decoder=decoder,
                beam_width=10
            )

            # Setup trainer
            trainer = get_trainer(
                model=combined_model,
                tokenized_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                evaluate_metric="wer",
                num_train_epochs=num_train_epochs,
                data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
                gradient_accumulation_steps=4,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
            )

            # Special case for movement pruning
            if method == "movement":
                # Initialize metadata for movement tracking
                for module in mg.model.modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                        if not hasattr(module, "metadata"):
                            module.metadata = {}
                        module.metadata["weight"] = {"stats": {"movement": torch.zeros_like(module.weight)}}
                
                # Add movement tracking callback
                trainer.add_callback(MovementTrackingCallback())
                
                # Do warm-up training to collect movement data
                print("Starting warm-up training for movement pruning...")
                trainer.train()
                print("Warm-up training complete.")
                
                # Reset trainer for actual training
                trainer = get_trainer(
                    model=combined_model,
                    tokenized_dataset=tokenized_dataset,
                    tokenizer=tokenizer,
                    evaluate_metric="wer",
                    num_train_epochs=num_train_epochs,
                    data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
                    gradient_accumulation_steps=4,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                )
            elif method == "snip":
                print("Preparing SNIP pruning with representative batch...")
                first_batch = next(iter(trainer.get_train_dataloader()))
                trainer.add_callback(SNIPCallback(representative_batch=first_batch))

            # Configure pruning
            pruning_config = {
                "weight": {
                    "sparsity": sparsity,
                    "method": method,
                    "scope": "local",
                    "granularity": "elementwise",
                },
                "activation": {
                    "sparsity": 0.0,
                    "method": "random",
                    "scope": "local",
                    "granularity": "elementwise",
                },
            }

            # Apply pruning
            print(f"Applying {method} pruning with {sparsity:.2f} sparsity...")
            mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

            # Get parameter statistics before training
            _, _ = report_trainable_parameters_analysis_pass(mg.model)
            total_params = sum(p.numel() for p in mg.model.parameters())
            trainable_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
            param_stats = {
                "total_params": total_params,
                "trainable_params": trainable_params
            }
            print(f"Parameter statistics after pruning:")
            print(f"Total parameters: {param_stats['total_params']}")
            print(f"Trainable parameters: {param_stats['trainable_params']}")
            
            # Compute and report actual sparsity achieved in mg.model
            actual_sparsity = compute_actual_sparsity(mg.model)
            print(f"Actual sparsity achieved: {actual_sparsity:.4f}")

            # <<-- Evaluate pruned model BEFORE fine tuning -->> 
            print("Evaluating pruned model BEFORE fine tuning...")
            initial_eval_results = trainer.evaluate()
            print(f"Initial WER: {initial_eval_results['eval_wer']:.4f}")
            
            # Train and evaluate (fine tuning)
            print("Starting training (fine tuning)...")
            trainer.train()

            # Get evaluation metrics after fine tuning
            print("Evaluating model AFTER fine tuning...")
            eval_results = trainer.evaluate()
            
            # Run runtime analysis
            print("Running runtime analysis...")
            _, runtime_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)

            # Combine results with both initial and final WERs and actual sparsity achieved
            method_results[sparsity] = {
                "initial_wer": initial_eval_results["eval_wer"],
                "wer": eval_results["eval_wer"],
                "loss": eval_results["eval_loss"],
                "runtime": eval_results["eval_runtime"],
                "samples_per_second": eval_results["eval_samples_per_second"],
                "steps_per_second": eval_results["eval_steps_per_second"],
                "average_wer": runtime_results["Average WER"],
                "average_latency": runtime_results["Average Latency"],
                "average_rtf": runtime_results["Average RTF"],
                "average_gpu_power": runtime_results["Average GPU Power Usage"],
                "inference_energy": runtime_results["Inference Energy Consumption"],
                "parameter_stats": param_stats,
                "actual_sparsity": actual_sparsity
            }

            print(f"\nResults for {method} at {sparsity:.2f} sparsity:")
            print(f"Initial WER (pre–fine tuning): {initial_eval_results['eval_wer']:.4f}")
            print(f"Final WER (post–fine tuning): {eval_results['eval_wer']:.4f}")
            print(f"Actual sparsity achieved: {actual_sparsity:.4f}")
            print(f"Average Latency: {runtime_results['Average Latency']:.2f} ms")
            print(f"Average RTF: {runtime_results['Average RTF']:.4f}")
            print(f"GPU Power Usage: {runtime_results['Average GPU Power Usage']:.2f} W")

        results[method] = method_results
        print(f"\nCompleted testing {method.upper()} pruning method")

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"pruning_comparison_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_file}")

    return results

def plot_results(results: Dict, save_dir: str = "plots"):
    """Create plots comparing different pruning methods."""
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style for better visualization
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Extract data for plotting
    methods = list(results.keys())
    sparsity_levels = list(results[methods[0]].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pruning Methods Comparison', fontsize=16)
    
    # Plot WER vs Sparsity (showing final WER; you could similarly plot initial WER if desired)
    ax = axes[0, 0]
    for method in methods:
        wer_values = [results[method][s]["wer"] for s in sparsity_levels]
        ax.plot(sparsity_levels, wer_values, marker='o', label=method)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('WER vs Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Plot Latency vs Sparsity
    ax = axes[0, 1]
    for method in methods:
        latency_values = [results[method][s]["average_latency"] for s in sparsity_levels]
        ax.plot(sparsity_levels, latency_values, marker='o', label=method)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Latency vs Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Plot RTF vs Sparsity
    ax = axes[1, 0]
    for method in methods:
        rtf_values = [results[method][s]["average_rtf"] for s in sparsity_levels]
        ax.plot(sparsity_levels, rtf_values, marker='o', label=method)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('RTF vs Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Plot GPU Power vs Sparsity
    ax = axes[1, 1]
    for method in methods:
        power_values = [results[method][s]["average_gpu_power"] for s in sparsity_levels]
        ax.plot(sparsity_levels, power_values, marker='o', label=method)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('GPU Power Usage (W)')
    ax.set_title('GPU Power vs Sparsity')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f'pruning_comparison_{timestamp}.png'))
    plt.close()
    
    # Create a separate plot for parameter counts
    plt.figure(figsize=(10, 6))
    for method in methods:
        param_values = [results[method][s]["parameter_stats"]["trainable_params"] for s in sparsity_levels]
        plt.plot(sparsity_levels, param_values, marker='o', label=method)
    plt.xlabel('Sparsity')
    plt.ylabel('Trainable Parameters')
    plt.title('Trainable Parameters vs Sparsity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'parameter_counts_{timestamp}.png'))
    plt.close()

def main():
    """Main function to run the pruning comparison experiment."""
    pruning_methods = ["random", "l1-norm", "movement", "snip", "hwpq"]
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Use 2 fine tuning epochs now
    results = run_pruning_experiment(
        pruning_methods=pruning_methods,
        sparsity_levels=sparsity_levels,
        num_train_epochs=2,
        save_results=True
    )
    
    plot_results(results)
    
    logger.info("\nExperiment Summary:")
    for method in pruning_methods:
        logger.info(f"\n{method} pruning results:")
        for s in sparsity_levels:
            method_results = results[method][s]
            logger.info(f"Sparsity {s}:")
            logger.info(f"  Initial WER (pre–fine tuning): {method_results['initial_wer']:.4f}")
            logger.info(f"  Final WER (post–fine tuning): {method_results['wer']:.4f}")
            logger.info(f"  Actual Sparsity Achieved: {method_results['actual_sparsity']:.4f}")
            logger.info(f"  Average Latency: {method_results['average_latency']:.2f} ms")
            logger.info(f"  Average RTF: {method_results['average_rtf']:.4f}")
            logger.info(f"  GPU Power Usage: {method_results['average_gpu_power']:.2f} W")
            logger.info(f"  Inference Energy: {method_results['inference_energy']:.2f} mWh")
            logger.info(f"  Total Parameters: {method_results['parameter_stats']['total_params']:,}")
            logger.info(f"  Trainable Parameters: {method_results['parameter_stats']['trainable_params']:,}")

if __name__ == "__main__":
    main()
