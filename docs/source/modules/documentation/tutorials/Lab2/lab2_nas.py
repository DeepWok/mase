#!/usr/bin/env python3
"""
Lab 2: Neural Architecture Search with Mase and Optuna
=======================================================
ELEC70109/EE9-AML3-10/EE9-AO25

This script implements the Lab 2 coursework tasks:
- Task 1: Compare RandomSampler, GridSampler, and TPESampler for NAS
- Task 2: Compression-aware NAS with quantization and pruning

Authors: Based on tutorials by Aaron Zhao and Pedro Gimenes
""" 
import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from transformers import AutoConfig, AutoModelForSequenceClassification

from chop import MaseGraph
from chop.nn.modules import Identity
from chop.pipelines import CompressionPipeline
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================

# Model checkpoints
CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"

# NAS Search Space (inspired by NAS-BERT paper)
SEARCH_SPACE = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [nn.Linear, Identity],
}

# Compression configuration for quantization
QUANTIZATION_CONFIG = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

# Pruning configuration
PRUNING_CONFIG = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}


# =============================================================================
# Data Loading
# =============================================================================

def get_dataset_and_tokenizer():
    """Load and tokenize the IMDb dataset."""
    print("\n" + "=" * 60)
    print("[STEP] Loading and tokenizing dataset...")
    print("=" * 60)
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Tokenizer checkpoint: {TOKENIZER_CHECKPOINT}")
    
    logger.info(f"Loading dataset: {DATASET_NAME} with tokenizer: {TOKENIZER_CHECKPOINT}")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=DATASET_NAME,
        checkpoint=TOKENIZER_CHECKPOINT,
        return_tokenizer=True,
    )
    
    print(f"  [OK] Dataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    return dataset, tokenizer


# =============================================================================
# Model Construction
# =============================================================================

def construct_model(trial: optuna.Trial, search_space: Dict) -> nn.Module:
    """
    Construct a BERT model with hyperparameters sampled by Optuna.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        search_space: Dictionary defining the search space
        
    Returns:
        Constructed model with sampled architecture
    """
    print(f"\n  [Trial {trial.number}] Constructing model...")
    
    config = AutoConfig.from_pretrained(CHECKPOINT)
    sampled_params = {}
    
    # Sample and set architectural hyperparameters
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        if param in search_space:
            chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
            chosen_value = search_space[param][chosen_idx]
            setattr(config, param, chosen_value)
            sampled_params[param] = chosen_value
    
    print(f"    Sampled hyperparameters: {sampled_params}")
    
    # Build model from config
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Optionally replace square linear layers with Identity (skip connections)
    # Only do this if linear_layer_choices is in the search space
    identity_count = 0
    if "linear_layer_choices" in search_space:
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                new_layer_cls = trial.suggest_categorical(
                    f"{name}_type",
                    search_space["linear_layer_choices"],
                )
                
                if new_layer_cls == nn.Linear:
                    continue
                elif new_layer_cls == Identity:
                    deepsetattr(model, name, Identity())
                    identity_count += 1
                else:
                    raise ValueError(f"Unknown layer type: {new_layer_cls}")
    
    num_params = count_parameters(model)
    print(f"    Model built: {num_params:,} parameters, {identity_count} Identity replacements")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Objective Functions
# =============================================================================

def create_objective(dataset, tokenizer, search_space: Dict, num_epochs: int = 3):
    """
    Create a standard NAS objective function for sampler comparison.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer instance
        search_space: Search space dictionary
        num_epochs: Number of training epochs per trial
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        print(f"\n" + "-" * 50)
        print(f"TRIAL {trial.number} STARTING")
        print("-" * 50)
        
        # Construct model with sampled hyperparameters
        model = construct_model(trial, search_space)
        
        logger.info(f"Trial {trial.number}: {count_parameters(model):,} parameters")
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [Trial {trial.number}] Training on {device} for {num_epochs} epoch(s)...")
        
        # Create trainer and train
        trainer = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=num_epochs,
        )
        
        trainer.train()
        print(f"  [Trial {trial.number}] Training complete. Evaluating...")
        
        eval_results = trainer.evaluate()
        
        accuracy = eval_results["eval_accuracy"]
        
        # Store model for later retrieval
        trial.set_user_attr("model", model.cpu())
        trial.set_user_attr("num_params", count_parameters(model))
        
        print(f"  [Trial {trial.number}] [OK] ACCURACY: {accuracy:.4f}")
        logger.info(f"Trial {trial.number}: accuracy = {accuracy:.4f}")
        
        return accuracy
    
    return objective


def create_compression_objective(
    dataset, 
    tokenizer, 
    search_space: Dict,
    num_epochs_pre: int = 2,
    num_epochs_post: int = 1,
    with_post_training: bool = True
):
    """
    Create a compression-aware objective function.
    
    The objective function:
    1. Constructs and trains the model for initial epochs
    2. Applies CompressionPipeline (quantization + pruning)
    3. Optionally continues training after compression
    
    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer instance
        search_space: Search space dictionary
        num_epochs_pre: Training epochs before compression
        num_epochs_post: Training epochs after compression (if enabled)
        with_post_training: Whether to train after compression
        
    Returns:
        Objective function for Optuna
    """
    pipe = CompressionPipeline()
    compression_config = {
        "quantize_transform_pass": QUANTIZATION_CONFIG,
        "prune_transform_pass": PRUNING_CONFIG,
    }
    
    def objective(trial: optuna.Trial) -> float:
        print(f"\n" + "-" * 50)
        print(f"COMPRESSION TRIAL {trial.number} STARTING")
        print("-" * 50)
        
        # Construct model
        model = construct_model(trial, search_space)
        logger.info(f"Trial {trial.number}: {count_parameters(model):,} parameters")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Phase 1: Initial training
        print(f"  [Trial {trial.number}] Phase 1: Pre-compression training ({num_epochs_pre} epochs) on {device}...")
        trainer = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=num_epochs_pre,
        )
        trainer.train()
        print(f"  [Trial {trial.number}] Phase 1 complete.")
        
        # Phase 2: Apply compression
        print(f"  [Trial {trial.number}] Phase 2: Applying compression (quantization + pruning)...")
        model.cpu()
        try:
            mg = MaseGraph(model)
            mg, _ = pipe(mg, copy.deepcopy(compression_config))
            print(f"  [Trial {trial.number}] [OK] Compression applied successfully.")
        except Exception as e:
            print(f"  [Trial {trial.number}] [FAIL] Compression FAILED: {e}")
            logger.warning(f"Trial {trial.number}: Compression failed: {e}")
            # If compression fails, return low score
            return 0.0
        
        # Phase 3: Post-compression training (optional)
        if with_post_training and num_epochs_post > 0:
            print(f"  [Trial {trial.number}] Phase 3: Post-compression training ({num_epochs_post} epochs)...")
            if torch.cuda.is_available():
                model.cuda()
            
            trainer = get_trainer(
                model=model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=num_epochs_post,
            )
            trainer.train()
            print(f"  [Trial {trial.number}] Phase 3 complete.")
        else:
            print(f"  [Trial {trial.number}] Phase 3: Skipped (no post-training).")
        
        # Evaluate final accuracy
        print(f"  [Trial {trial.number}] Evaluating final accuracy...")
        if torch.cuda.is_available():
            model.cuda()
        
        trainer = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=0,  # Just for evaluation
        )
        eval_results = trainer.evaluate()
        
        accuracy = eval_results["eval_accuracy"]
        
        trial.set_user_attr("model", model.cpu())
        trial.set_user_attr("num_params", count_parameters(model))
        
        print(f"  [Trial {trial.number}] [OK] ACCURACY: {accuracy:.4f} (compression-aware)")
        logger.info(f"Trial {trial.number}: accuracy = {accuracy:.4f} (compression-aware)")
        
        return accuracy
    
    return objective


# =============================================================================
# Experiment Functions
# =============================================================================

def compute_max_accuracy_curve(study: optuna.Study) -> List[float]:
    """
    Compute the maximum achieved accuracy at each trial point.
    
    Args:
        study: Completed Optuna study
        
    Returns:
        List of max accuracies up to each trial
    """
    max_accuracies = []
    current_max = 0.0
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            current_max = max(current_max, trial.value)
        max_accuracies.append(current_max)
    
    return max_accuracies


def run_sampler_comparison(
    dataset, 
    tokenizer, 
    n_trials: int = 20,
    num_epochs: int = 3,
    output_dir: Path = Path(".")
) -> Dict[str, List[float]]:
    """
    Task 1: Run NAS with different samplers and compare performance.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer instance
        n_trials: Number of trials per sampler
        num_epochs: Training epochs per trial
        output_dir: Directory for saving results
        
    Returns:
        Dictionary mapping sampler names to their max accuracy curves
    """
    logger.info("=" * 60)
    logger.info("TASK 1: Sampler Comparison")
    logger.info("=" * 60)
    
    # Build full search space including linear layer types
    full_search_space = copy.deepcopy(SEARCH_SPACE)
    
    # Create objective function
    objective = create_objective(dataset, tokenizer, full_search_space, num_epochs)
    
    results = {}
    samplers = {
        "Random": RandomSampler(),
        "TPE": TPESampler(),
    }
    
    # Note: GridSampler requires explicit search space and may not complete in n_trials
    # if the search space is too large. We'll handle it separately.
    
    for sampler_name, sampler in samplers.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running {sampler_name}Sampler with {n_trials} trials")
        logger.info(f"{'=' * 40}")
        
        study = optuna.create_study(
            direction="maximize",
            study_name=f"bert-nas-{sampler_name.lower()}",
            sampler=sampler,
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=60 * 60 * 24,  # 24 hour timeout
            show_progress_bar=True,
        )
        
        # Compute max accuracy curve
        results[sampler_name] = compute_max_accuracy_curve(study)
        
        logger.info(f"{sampler_name}: Best accuracy = {study.best_value:.4f}")
        logger.info(f"{sampler_name}: Best params = {study.best_params}")
    
    # For GridSampler, we need a STATIC search space. The linear layer type
    # choices are dynamic (depend on num_layers), so we exclude them for Grid.
    logger.info(f"\n{'=' * 40}")
    logger.info("Running GridSampler with {n_trials} trials")
    logger.info("NOTE: Grid search excludes linear layer types (they're dynamic)")
    logger.info(f"{'=' * 40}")
    
    # Create grid search space WITHOUT linear layer types
    grid_search_space = {
        "num_layers": list(range(len(SEARCH_SPACE["num_layers"]))),
        "num_heads": list(range(len(SEARCH_SPACE["num_heads"]))),
        "hidden_size": list(range(len(SEARCH_SPACE["hidden_size"]))),
        "intermediate_size": list(range(len(SEARCH_SPACE["intermediate_size"]))),
    }
    
    # Create a separate objective that doesn't use linear layer choices
    grid_only_search_space = {k: v for k, v in SEARCH_SPACE.items() if k != "linear_layer_choices"}
    grid_objective = create_objective(dataset, tokenizer, grid_only_search_space, num_epochs)
    
    grid_sampler = GridSampler(grid_search_space)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="bert-nas-grid",
        sampler=grid_sampler,
    )
    
    study.optimize(
        grid_objective,
        n_trials=n_trials,
        timeout=60 * 60 * 24,
        show_progress_bar=True,
    )
    
    results["Grid"] = compute_max_accuracy_curve(study)
    
    logger.info(f"Grid: Best accuracy = {study.best_value:.4f}")
    
    return results


def run_compression_aware_search(
    dataset,
    tokenizer,
    n_trials: int = 15,
    num_epochs_pre: int = 2,
    num_epochs_post: int = 1,
    best_baseline_accuracy: float = 0.0,
    output_dir: Path = Path(".")
) -> Dict[str, List[float]]:
    """
    Task 2: Run compression-aware NAS with and without post-compression training.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer instance
        n_trials: Number of trials
        num_epochs_pre: Training epochs before compression
        num_epochs_post: Training epochs after compression
        best_baseline_accuracy: Best accuracy from Task 1 (for baseline curve)
        output_dir: Directory for saving results
        
    Returns:
        Dictionary with curves for each variant
    """
    logger.info("=" * 60)
    logger.info("TASK 2: Compression-Aware NAS")
    logger.info("=" * 60)
    
    full_search_space = copy.deepcopy(SEARCH_SPACE)
    results = {}
    
    # Use TPE sampler (typically best from Task 1)
    sampler = TPESampler()
    
    # Variant 1: Without post-compression training
    logger.info(f"\n{'=' * 40}")
    logger.info("Running compression-aware search WITHOUT post-training")
    logger.info(f"{'=' * 40}")
    
    objective_no_post = create_compression_objective(
        dataset, tokenizer, full_search_space,
        num_epochs_pre=num_epochs_pre,
        num_epochs_post=0,
        with_post_training=False
    )
    
    study_no_post = optuna.create_study(
        direction="maximize",
        study_name="bert-nas-compression-no-post",
        sampler=TPESampler(),
    )
    
    study_no_post.optimize(
        objective_no_post,
        n_trials=n_trials,
        timeout=60 * 60 * 24,
        show_progress_bar=True,
    )
    
    results["Compression (no post-training)"] = compute_max_accuracy_curve(study_no_post)
    logger.info(f"Without post-training: Best accuracy = {study_no_post.best_value:.4f}")
    
    # Variant 2: With post-compression training
    logger.info(f"\n{'=' * 40}")
    logger.info("Running compression-aware search WITH post-training")
    logger.info(f"{'=' * 40}")
    
    objective_with_post = create_compression_objective(
        dataset, tokenizer, full_search_space,
        num_epochs_pre=num_epochs_pre,
        num_epochs_post=num_epochs_post,
        with_post_training=True
    )
    
    study_with_post = optuna.create_study(
        direction="maximize",
        study_name="bert-nas-compression-with-post",
        sampler=TPESampler(),
    )
    
    study_with_post.optimize(
        objective_with_post,
        n_trials=n_trials,
        timeout=60 * 60 * 24,
        show_progress_bar=True,
    )
    
    results["Compression (with post-training)"] = compute_max_accuracy_curve(study_with_post)
    logger.info(f"With post-training: Best accuracy = {study_with_post.best_value:.4f}")
    
    # Add baseline curve (constant at best Task 1 accuracy)
    results["Baseline (no compression)"] = [best_baseline_accuracy] * n_trials
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_sampler_comparison(results: Dict[str, List[float]], output_path: Path):
    """
    Plot Task 1 results: Max accuracy vs. number of trials for each sampler.
    
    Args:
        results: Dictionary mapping sampler names to max accuracy curves
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = {"Random": "#1f77b4", "Grid": "#ff7f0e", "TPE": "#2ca02c"}
    markers = {"Random": "o", "Grid": "s", "TPE": "^"}
    
    for sampler_name, accuracies in results.items():
        trials = list(range(1, len(accuracies) + 1))
        plt.plot(
            trials, 
            accuracies, 
            label=f"{sampler_name}Sampler",
            color=colors.get(sampler_name, None),
            marker=markers.get(sampler_name, "o"),
            markevery=max(1, len(trials) // 10),  # Mark every ~10% of points
            linewidth=2,
            markersize=8,
        )
    
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Accuracy Achieved", fontsize=12)
    plt.title("Neural Architecture Search: Sampler Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved sampler comparison plot to {output_path}")
    plt.close()


def plot_compression_comparison(results: Dict[str, List[float]], output_path: Path):
    """
    Plot Task 2 results: Comparison of compression-aware search variants.
    
    Args:
        results: Dictionary with accuracy curves for each variant
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        "Baseline (no compression)": "#1f77b4",
        "Compression (no post-training)": "#ff7f0e",
        "Compression (with post-training)": "#2ca02c",
    }
    linestyles = {
        "Baseline (no compression)": "--",
        "Compression (no post-training)": "-",
        "Compression (with post-training)": "-",
    }
    markers = {
        "Baseline (no compression)": None,
        "Compression (no post-training)": "s",
        "Compression (with post-training)": "^",
    }
    
    for variant_name, accuracies in results.items():
        trials = list(range(1, len(accuracies) + 1))
        plt.plot(
            trials,
            accuracies,
            label=variant_name,
            color=colors.get(variant_name, None),
            linestyle=linestyles.get(variant_name, "-"),
            marker=markers.get(variant_name, None),
            markevery=max(1, len(trials) // 10),
            linewidth=2,
            markersize=8,
        )
    
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Accuracy Achieved", fontsize=12)
    plt.title("Compression-Aware Neural Architecture Search", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved compression comparison plot to {output_path}")
    plt.close()


# =============================================================================
# Data Export Functions
# =============================================================================

def save_results_to_json(
    task1_results: Dict[str, List[float]],
    task2_results: Dict[str, List[float]],
    config: dict,
    output_path: Path
):
    """
    Save experiment results to JSON for later analysis or re-plotting.
    
    Args:
        task1_results: Dictionary of sampler name -> max accuracy curves
        task2_results: Dictionary of variant name -> max accuracy curves
        config: Experiment configuration parameters
        output_path: Path to save the JSON file
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "task1_sampler_comparison": {
            name: {
                "max_accuracy_curve": accuracies,
                "best_accuracy": max(accuracies) if accuracies else 0.0,
                "num_trials": len(accuracies),
            }
            for name, accuracies in task1_results.items()
        } if task1_results else None,
        "task2_compression_aware": {
            name: {
                "max_accuracy_curve": accuracies,
                "best_accuracy": max(accuracies) if accuracies else 0.0,
                "num_trials": len(accuracies),
            }
            for name, accuracies in task2_results.items()
        } if task2_results else None,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_path}")
    logger.info(f"Saved results to {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lab 2: Neural Architecture Search with Mase and Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (1 trial, 1 epoch) to verify everything works
  python lab2_nas.py --test
  
  # Run full experiment (Task 1 + Task 2)
  python lab2_nas.py --n-trials 20 --epochs 3
  
  # Run only Task 1 (sampler comparison)
  python lab2_nas.py --task1-only --n-trials 20
  
  # Run only Task 2 (requires baseline from Task 1)
  python lab2_nas.py --task2-only --baseline-accuracy 0.85
        """
    )
    
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of trials for Task 1 samplers (default: 20)"
    )
    parser.add_argument(
        "--n-trials-compression", type=int, default=15,
        help="Number of trials for Task 2 compression search (default: 15)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs per trial in Task 1 (default: 3)"
    )
    parser.add_argument(
        "--epochs-pre", type=int, default=2,
        help="Pre-compression training epochs in Task 2 (default: 2)"
    )
    parser.add_argument(
        "--epochs-post", type=int, default=1,
        help="Post-compression training epochs in Task 2 (default: 1)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Output directory for plots and results (default: current directory)"
    )
    parser.add_argument(
        "--task1-only", action="store_true",
        help="Run only Task 1 (sampler comparison)"
    )
    parser.add_argument(
        "--task2-only", action="store_true",
        help="Run only Task 2 (compression-aware search)"
    )
    parser.add_argument(
        "--baseline-accuracy", type=float, default=0.85,
        help="Baseline accuracy for Task 2 (when running --task2-only)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode: 1 trial, 1 epoch for quick validation"
    )
    
    args = parser.parse_args()
    
    # Handle --test mode: override all trials and epochs to 1
    if args.test:
        print("\n" + "!" * 60)
        print("!!! TEST MODE ENABLED !!!")
        print("!!! Running with 1 trial and 1 epoch for quick validation !!!")
        print("!" * 60)
        args.n_trials = 1
        args.n_trials_compression = 1
        args.epochs = 1
        args.epochs_pre = 1
        args.epochs_post = 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Lab 2: Neural Architecture Search Configuration")
    logger.info("=" * 60)
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Task 1 trials: {args.n_trials}")
    logger.info(f"Task 2 trials: {args.n_trials_compression}")
    logger.info(f"Task 1 epochs: {args.epochs}")
    logger.info(f"Task 2 epochs (pre/post): {args.epochs_pre}/{args.epochs_post}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load dataset
    dataset, tokenizer = get_dataset_and_tokenizer()
    
    best_baseline_accuracy = args.baseline_accuracy
    task1_results = {}
    task2_results = {}
    
    # Task 1: Sampler Comparison
    if not args.task2_only:
        task1_results = run_sampler_comparison(
            dataset, tokenizer,
            n_trials=args.n_trials,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
        )
        
        # Find best baseline accuracy for Task 2
        for sampler_name, accuracies in task1_results.items():
            if accuracies:
                best_baseline_accuracy = max(best_baseline_accuracy, max(accuracies))
        
        # Plot Task 1 results
        plot_sampler_comparison(
            task1_results,
            args.output_dir / "sampler_comparison.png"
        )
        
        logger.info(f"\nTask 1 Complete! Best baseline accuracy: {best_baseline_accuracy:.4f}")
    
    # Task 2: Compression-Aware Search
    if not args.task1_only:
        task2_results = run_compression_aware_search(
            dataset, tokenizer,
            n_trials=args.n_trials_compression,
            num_epochs_pre=args.epochs_pre,
            num_epochs_post=args.epochs_post,
            best_baseline_accuracy=best_baseline_accuracy,
            output_dir=args.output_dir,
        )
        
        # Plot Task 2 results
        plot_compression_comparison(
            task2_results,
            args.output_dir / "compression_comparison.png"
        )
        
        logger.info("\nTask 2 Complete!")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    
    if task1_results:
        logger.info("\nTask 1 Results (Best accuracy per sampler):")
        for sampler_name, accuracies in task1_results.items():
            if accuracies:
                logger.info(f"  {sampler_name}Sampler: {max(accuracies):.4f}")
    
    if task2_results:
        logger.info("\nTask 2 Results:")
        for variant_name, accuracies in task2_results.items():
            if accuracies:
                logger.info(f"  {variant_name}: {max(accuracies):.4f}")
    
    # Save results to JSON for later analysis or re-plotting
    experiment_config = {
        "n_trials": args.n_trials,
        "n_trials_compression": args.n_trials_compression,
        "epochs": args.epochs,
        "epochs_pre": args.epochs_pre,
        "epochs_post": args.epochs_post,
        "checkpoint": CHECKPOINT,
        "tokenizer_checkpoint": TOKENIZER_CHECKPOINT,
        "dataset": DATASET_NAME,
        "test_mode": args.test if hasattr(args, 'test') else False,
    }
    
    save_results_to_json(
        task1_results,
        task2_results,
        experiment_config,
        args.output_dir / "results.json"
    )
    
    logger.info(f"\nPlots saved to: {args.output_dir}")
    logger.info(f"Results data saved to: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()