#!/usr/bin/env python3
"""
Lab 1: Quantization and Pruning Experiments for BERT

This script implements the Lab 1 experiments:
- Task 1: Explore quantization bit widths (4-32) and plot PTQ vs QAT accuracy
- Task 2: Explore sparsity levels (0.1-0.9) with Random vs L1-Norm pruning

Usage:
    python lab1_complete.py [--test]  # --test runs quick validation with fewer configs
"""

import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_trainer


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def setup_dataset(checkpoint: str = "prajjwal1/bert-tiny", dataset_name: str = "imdb"):
    """Load and tokenize dataset."""
    print("Loading and tokenizing dataset...")
    
    raw_datasets = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    dataset = raw_datasets.map(tokenize_function, batched=True)
    return dataset, tokenizer


def create_fresh_model(checkpoint: str = "prajjwal1/bert-tiny"):
    """Create a fresh model and MaseGraph."""
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.config.problem_type = "single_label_classification"
    
    mg = MaseGraph(
        model,
        hf_input_names=["input_ids", "attention_mask", "labels"],
    )
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg)
    return mg


def finetune_model(mg, dataset, tokenizer, num_epochs=1):
    """Fine-tune the model with LoRA and return fused model."""
    # Freeze embeddings
    for param in mg.model.bert.embeddings.parameters():
        param.requires_grad = False
    
    # Insert LoRA adapters
    mg, _ = passes.insert_lora_adapter_transform_pass(
        mg,
        pass_args={"rank": 6, "alpha": 1.0, "dropout": 0.5},
    )
    
    # Train
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=num_epochs,
    )
    trainer.train()
    
    # Fuse LoRA weights
    mg, _ = passes.fuse_lora_weights_transform_pass(mg)
    
    # Evaluate and export
    eval_results = trainer.evaluate()
    base_acc = eval_results['eval_accuracy']
    print(f"Base model accuracy after LoRA: {base_acc:.4f}")
    
    # Export for reuse
    mg.export(f"{Path.home()}/lab1_base_model")
    
    return mg, base_acc


def load_finetuned_model():
    """Load the pre-finetuned model checkpoint."""
    checkpoint_path = f"{Path.home()}/lab1_base_model"
    if not Path(f"{checkpoint_path}.pt").exists():
        return None
    mg = MaseGraph.from_checkpoint(checkpoint_path)
    return mg


def task1_quantization_sweep(dataset, tokenizer, checkpoint, bit_widths, num_qat_epochs=1):
    """
    Task 1: Sweep quantization bit widths and compare PTQ vs QAT accuracy.
    
    Returns dict with 'widths', 'ptq_acc', 'qat_acc' lists.
    """
    print_section("Task 1: Quantization Bit Width Sweep")
    
    results = {
        'widths': [],
        'ptq_acc': [],
        'qat_acc': [],
    }
    
    for width in bit_widths:
        print(f"\n--- Testing bit width: {width} ---")
        
        # Load fresh model from checkpoint
        mg = load_finetuned_model()
        if mg is None:
            print("Error: Base model checkpoint not found. Running fine-tuning first...")
            mg = create_fresh_model(checkpoint)
            mg, _ = finetune_model(mg, dataset, tokenizer)
            mg = load_finetuned_model()
        
        # Quantization config
        frac_width = width // 2
        quantization_config = {
            "by": "type",
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "integer",
                    "data_in_width": width,
                    "data_in_frac_width": frac_width,
                    "weight_width": width,
                    "weight_frac_width": frac_width,
                    "bias_width": width,
                    "bias_frac_width": frac_width,
                }
            },
        }
        
        # Apply quantization (PTQ)
        mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
        
        # Evaluate PTQ
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=num_qat_epochs,
        )
        ptq_results = trainer.evaluate()
        ptq_acc = ptq_results['eval_accuracy']
        print(f"  PTQ accuracy (width={width}): {ptq_acc:.4f}")
        
        # Run QAT
        trainer.train()
        qat_results = trainer.evaluate()
        qat_acc = qat_results['eval_accuracy']
        print(f"  QAT accuracy (width={width}): {qat_acc:.4f}")
        
        results['widths'].append(width)
        results['ptq_acc'].append(ptq_acc)
        results['qat_acc'].append(qat_acc)
    
    return results


def task2_pruning_sweep(dataset, tokenizer, checkpoint, best_width, sparsities, num_finetune_epochs=1):
    """
    Task 2: Sweep sparsity levels and compare Random vs L1-Norm pruning.
    
    Returns dict with 'sparsities', 'random_acc', 'l1_acc' lists.
    """
    print_section("Task 2: Pruning Sparsity Sweep")
    
    results = {
        'sparsities': [],
        'random_acc': [],
        'l1_acc': [],
    }
    
    for sparsity in sparsities:
        print(f"\n--- Testing sparsity: {sparsity:.1f} ---")
        results['sparsities'].append(sparsity)
        
        for method in ['random', 'l1-norm']:
            print(f"  Method: {method}")
            
            # Load fresh model from checkpoint
            mg = load_finetuned_model()
            
            # Apply best quantization first
            frac_width = best_width // 2
            quantization_config = {
                "by": "type",
                "default": {"config": {"name": None}},
                "linear": {
                    "config": {
                        "name": "integer",
                        "data_in_width": best_width,
                        "data_in_frac_width": frac_width,
                        "weight_width": best_width,
                        "weight_frac_width": frac_width,
                        "bias_width": best_width,
                        "bias_frac_width": frac_width,
                    }
                },
            }
            mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
            
            # QAT training
            trainer = get_trainer(
                model=mg.model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=1,
            )
            trainer.train()
            
            # Pruning config
            pruning_config = {
                "weight": {
                    "sparsity": sparsity,
                    "method": method,
                    "scope": "local",
                },
                "activation": {
                    "sparsity": sparsity,
                    "method": method,
                    "scope": "local",
                },
            }
            
            # Apply pruning
            mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
            
            # Fine-tune and evaluate
            trainer = get_trainer(
                model=mg.model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=num_finetune_epochs,
            )
            trainer.train()
            eval_results = trainer.evaluate()
            acc = eval_results['eval_accuracy']
            print(f"    Accuracy: {acc:.4f}")
            
            if method == 'random':
                results['random_acc'].append(acc)
            else:
                results['l1_acc'].append(acc)
    
    return results


def plot_quantization_results(results, output_path="task1_quantization.png"):
    """Plot Task 1: Quantization bit width vs accuracy."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['widths'], results['ptq_acc'], 'o-', label='PTQ', linewidth=2, markersize=8)
    plt.plot(results['widths'], results['qat_acc'], 's-', label='QAT', linewidth=2, markersize=8)
    
    plt.xlabel('Fixed-Point Bit Width', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Task 1: Effect of Quantization Bit Width on Accuracy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(results['widths'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved quantization plot to: {output_path}")
    plt.close()


def plot_pruning_results(results, output_path="task2_pruning.png"):
    """Plot Task 2: Sparsity vs accuracy for different pruning methods."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['sparsities'], results['random_acc'], 'o-', label='Random', linewidth=2, markersize=8)
    plt.plot(results['sparsities'], results['l1_acc'], 's-', label='L1-Norm', linewidth=2, markersize=8)
    
    plt.xlabel('Sparsity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Task 2: Effect of Pruning Sparsity on Accuracy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved pruning plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lab 1: Quantization and Pruning Experiments")
    parser.add_argument("--test", action="store_true", help="Run quick test with fewer configurations")
    args = parser.parse_args()
    
    print_section("Lab 1: Quantization and Pruning Experiments")
    
    # Configuration
    checkpoint = "prajjwal1/bert-tiny"
    dataset_name = "imdb"
    output_dir = Path("lab1_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Model: {checkpoint}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Test mode: {args.test}")
    
    # Experiment configuration
    if args.test:
        bit_widths = [8, 16]
        sparsities = [0.3, 0.5, 0.7]
        num_qat_epochs = 1
        num_finetune_epochs = 1
    else:
        bit_widths = [4, 8, 12, 16, 20, 24, 28, 32]
        sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        num_qat_epochs = 1
        num_finetune_epochs = 1
    
    # Setup dataset
    dataset, tokenizer = setup_dataset(checkpoint, dataset_name)
    
    # Check if base model exists, if not create it
    if load_finetuned_model() is None:
        print("\n--- Creating base fine-tuned model ---")
        mg = create_fresh_model(checkpoint)
        mg, base_acc = finetune_model(mg, dataset, tokenizer)
    else:
        print("\n--- Using existing base model checkpoint ---")
        mg = load_finetuned_model()
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
        )
        base_acc = trainer.evaluate()['eval_accuracy']
        print(f"Base model accuracy: {base_acc:.4f}")
    
    # Task 1: Quantization sweep
    quant_results = task1_quantization_sweep(
        dataset, tokenizer, checkpoint,
        bit_widths=bit_widths,
        num_qat_epochs=num_qat_epochs
    )
    
    # Find best quantized model
    best_qat_idx = quant_results['qat_acc'].index(max(quant_results['qat_acc']))
    best_width = quant_results['widths'][best_qat_idx]
    print(f"\nBest quantization width: {best_width} (QAT acc: {quant_results['qat_acc'][best_qat_idx]:.4f})")
    
    # Task 2: Pruning sweep
    prune_results = task2_pruning_sweep(
        dataset, tokenizer, checkpoint,
        best_width=best_width,
        sparsities=sparsities,
        num_finetune_epochs=num_finetune_epochs
    )
    
    # Generate plots
    plot_quantization_results(quant_results, output_dir / "task1_quantization.png")
    plot_pruning_results(prune_results, output_dir / "task2_pruning.png")
    
    # Save results to JSON
    all_results = {
        'base_accuracy': base_acc,
        'quantization': quant_results,
        'pruning': prune_results,
        'best_quantization_width': best_width,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {output_dir / 'results.json'}")
    
    print_section("Lab 1 Complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - task1_quantization.png")
    print(f"  - task2_pruning.png")
    print(f"  - results.json")


if __name__ == "__main__":
    main()
