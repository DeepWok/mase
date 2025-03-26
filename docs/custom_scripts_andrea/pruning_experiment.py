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
from chop.passes import add_movement_metadata_analysis_pass
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

# Take a larger subset of the dataset for training
train_subset = tokenized_dataset["train"].select(range(500))  # Use 500 samples
test_subset = tokenized_dataset["test"].select(range(100))     # Use 100 samples
tokenized_dataset = DatasetDict({
    "train": train_subset,
    "test": test_subset
})

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

# -------------------------------
# 2. Import dataset
# -------------------------------

batch_size = 8  # Increased batch size since we have more data
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=checkpoint,
    num_workers=0,
    processor=processor
)
data_module.setup()

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

def count_nonzero_parameters(model):
    """Count the actual non-zero parameters in the model"""
    total_params = 0
    nonzero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                # Get the actual weight after parametrization
                weight = module.parametrizations.weight[0](module.weight)
            else:
                weight = module.weight
            
            # Count total parameters
            total_params += weight.numel()
            
            # Count non-zero parameters
            nonzero_params += (weight != 0).sum().item()
                
    return total_params, nonzero_params

def print_parameter_count(model, description):
    """Helper function to count and print parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Also count non-zero parameters
    total, nonzero = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    
    print(f"\n===== {description} =====")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total weight parameters: {total:,}")
    print(f"Non-zero weight parameters: {nonzero:,}")
    print(f"Actual sparsity: {sparsity:.2%}")
    
    return total_params, nonzero, sparsity

def compute_actual_sparsity(model: nn.Module) -> float:
    """Compute the actual sparsity of the model after pruning parametrization."""
    total_params = 0
    total_zeros = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                # Get the actual weight after parametrization
                weight = module.parametrizations.weight[0](module.weight)
            else:
                weight = module.weight
            
            # Count total parameters
            total_params += weight.numel()
            
            # Count zero parameters
            total_zeros += (weight == 0).sum().item()
    
    return total_zeros / total_params if total_params > 0 else 0.0

def run_pruning_experiment(
    pruning_methods: List[str] = ["random", "l1-norm", "movement", "snip", "hwpq"],
    sparsity_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    num_train_epochs: int = 2,
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

            # Add appropriate callbacks based on pruning method
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
            # Calculate total parameters manually since the analysis pass only prints
            total_params = sum(p.numel() for p in mg.model.parameters())
            trainable_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
            param_stats = {
                "total_params": total_params,
                "trainable_params": trainable_params
            }
            print(f"Parameter statistics after pruning:")
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

            # Evaluate before fine-tuning
            print("Evaluating model immediately after pruning...")
            pre_finetune_results = trainer.evaluate()
            pre_finetune_wer = pre_finetune_results["eval_wer"]
            print(f"Pre-fine-tuning WER: {pre_finetune_wer:.4f}")

            # Train and evaluate
            print(f"Starting fine-tuning with {method} pruning at {sparsity} sparsity...")
            trainer.train()

            # Get evaluation metrics after fine-tuning
            eval_results = trainer.evaluate()
            
            # Run runtime analysis
            print("Running runtime analysis...")
            _, runtime_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)

            # Calculate actual sparsity
            actual_sparsity = compute_actual_sparsity(mg.model)
            print(f"Target sparsity: {sparsity:.2%}")
            print(f"Actual sparsity: {actual_sparsity:.2%}")
            print(f"Difference: {(actual_sparsity - sparsity):.2%}")

            # Combine results
            method_results[sparsity] = {
                "wer": eval_results["eval_wer"],
                "pre_finetune_wer": pre_finetune_wer,
                "wer_improvement": pre_finetune_wer - eval_results["eval_wer"],
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
                "actual_sparsity": actual_sparsity,
                "sparsity_difference": actual_sparsity - sparsity
            }

            print(f"\nResults for {method} at {sparsity:.2f} sparsity:")
            print(f"Pre-fine-tuning WER: {pre_finetune_wer:.4f}")
            print(f"Post-fine-tuning WER: {eval_results['eval_wer']:.4f}")
            print(f"WER improvement: {(pre_finetune_wer - eval_results['eval_wer']):.4f}")
            print(f"Average Latency: {runtime_results['Average Latency']:.2f} ms")
            print(f"Average RTF: {runtime_results['Average RTF']:.4f}")
            print(f"GPU Power Usage: {runtime_results['Average GPU Power Usage']:.2f} W")
            print(f"Actual sparsity: {actual_sparsity:.2%}")

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
    
    # Plot WER vs Sparsity (pre and post fine-tuning)
    ax = axes[0, 0]
    for method in methods:
        # Plot pre-fine-tuning WER
        pre_wer_values = [results[method][sparsity]["pre_finetune_wer"] for sparsity in sparsity_levels]
        ax.plot(sparsity_levels, pre_wer_values, marker='o', linestyle='--', label=f'{method} (pre)')
        
        # Plot post-fine-tuning WER
        post_wer_values = [results[method][sparsity]["wer"] for sparsity in sparsity_levels]
        ax.plot(sparsity_levels, post_wer_values, marker='o', label=f'{method} (post)')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('WER vs Sparsity (Pre/Post Fine-tuning)')
    ax.grid(True)
    ax.legend()
    
    # Plot Actual vs Target Sparsity
    ax = axes[0, 1]
    for method in methods:
        actual_sparsity = [results[method][sparsity]["actual_sparsity"] for sparsity in sparsity_levels]
        ax.plot(sparsity_levels, actual_sparsity, marker='o', label=method)
    # Add diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Match')
    ax.set_xlabel('Target Sparsity')
    ax.set_ylabel('Actual Sparsity')
    ax.set_title('Actual vs Target Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Plot WER Improvement
    ax = axes[1, 0]
    for method in methods:
        improvement = [results[method][sparsity]["wer_improvement"] for sparsity in sparsity_levels]
        ax.plot(sparsity_levels, improvement, marker='o', label=method)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('WER Improvement')
    ax.set_title('WER Improvement vs Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Plot Sparsity Difference
    ax = axes[1, 1]
    for method in methods:
        diff = [results[method][sparsity]["sparsity_difference"] for sparsity in sparsity_levels]
        ax.plot(sparsity_levels, diff, marker='o', label=method)
    ax.set_xlabel('Target Sparsity')
    ax.set_ylabel('Sparsity Difference (Actual - Target)')
    ax.set_title('Sparsity Difference vs Target Sparsity')
    ax.grid(True)
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f'pruning_comparison_{timestamp}.png'))
    plt.close()
    
    # Create a separate plot for parameter counts
    plt.figure(figsize=(10, 6))
    for method in methods:
        param_values = [results[method][sparsity]["parameter_stats"]["trainable_params"] for sparsity in sparsity_levels]
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
    # Define pruning methods and sparsity levels to test
    pruning_methods = ["random", "l1-norm", "movement", "snip", "hwpq"]
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Updated sparsity levels
    
    # Run experiment with 1 fine-tuning epoch
    results = run_pruning_experiment(
        pruning_methods=pruning_methods,
        sparsity_levels=sparsity_levels,
        num_train_epochs=1,  # Changed to 1 epoch
        save_results=True
    )
    
    # Create plots
    plot_results(results)
    
    # Print summary
    logger.info("\nExperiment Summary:")
    for method in pruning_methods:
        logger.info(f"\n{method} pruning results:")
        for sparsity in sparsity_levels:
            method_results = results[method][sparsity]
            logger.info(f"Sparsity {sparsity}:")
            logger.info(f"  WER: {method_results['wer']:.4f}")
            logger.info(f"  Average Latency: {method_results['average_latency']:.2f} ms")
            logger.info(f"  Average RTF: {method_results['average_rtf']:.4f}")
            logger.info(f"  GPU Power Usage: {method_results['average_gpu_power']:.2f} W")
            logger.info(f"  Inference Energy: {method_results['inference_energy']:.2f} mWh")
            logger.info(f"  Total Parameters: {method_results['parameter_stats']['total_params']:,}")
            logger.info(f"  Trainable Parameters: {method_results['parameter_stats']['trainable_params']:,}")

if __name__ == "__main__":
    main() 
