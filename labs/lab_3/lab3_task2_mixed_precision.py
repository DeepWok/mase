#!/usr/bin/env python3
"""
Tutorial 6 - Task 2: Mixed Precision Search with All Supported Precisions

Goal:
- Extend the search to consider ALL supported precisions for Linear layers in MASE
- For each nn.Linear layer, choose from: [nn.Linear, LinearInteger, LinearMinifloatIEEE,
  LinearMinifloatDenorm, LinearLog, LinearBlockFP, LinearBlockMinifloat, LinearBlockLog, LinearBinary]
- If a quantized type is chosen, sample the appropriate hyperparameters for that precision
- Run Optuna search and plot: trial index vs best accuracy so far
- Plot one curve for each precision to compare their performance

Usage:
    python multi_precision_search.py --sampler random --n_trials 20
    python multi_precision_search.py --sampler tpe --n_trials 20
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import numpy as np
import dill
import torch
import torch.nn as nn

import optuna
from optuna.samplers import RandomSampler, TPESampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
)


# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"
SEED = 42

# All supported quantized layer types
QUANTIZED_LAYER_CLASSES = [
    LinearInteger,
    LinearMinifloatIEEE,
    LinearMinifloatDenorm,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
]

# Full search space: full precision + all quantized types
LINEAR_LAYER_CHOICES = [nn.Linear] + QUANTIZED_LAYER_CLASSES

# Search space choices for different precision types
WIDTH_CHOICES = [8, 16, 32]
FRAC_WIDTH_CHOICES = [2, 4, 8]
EXPONENT_WIDTH_CHOICES = [3, 4, 5]
EXPONENT_BIAS_CHOICES = [3, 7, 15]
BLOCK_SIZE_CHOICES = [8, 16, 32]


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(output_dir: Path, run_name: str) -> logging.Logger:
    """Set up logging to both file and console."""

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    log_file = output_dir / f"{run_name}.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# Base Model Loading
# ============================================================================
def load_base_model(base_model_path: str | None, checkpoint: str) -> nn.Module:
    """
    Load base model from previous NAS stage or from checkpoint.
    """
    if base_model_path is None:
        return AutoModelForSequenceClassification.from_pretrained(checkpoint)

    p = Path(base_model_path)
    if not p.exists():
        raise FileNotFoundError(f"Base model path does not exist: {p}")

    if p.suffix == ".pt":
        payload = torch.load(p, map_location="cpu")
        ckpt = payload.get("checkpoint", checkpoint)
        best_params = payload.get("best_params", None)

        if best_params is not None and all(
            k in best_params for k in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]
        ):
            config = AutoConfig.from_pretrained(ckpt)
            config.num_hidden_layers = best_params["num_layers"]
            config.num_attention_heads = best_params["num_heads"]
            config.hidden_size = best_params["hidden_size"]
            config.intermediate_size = best_params["intermediate_size"]
            model = AutoModelForSequenceClassification.from_config(config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt)

        model.load_state_dict(payload["state_dict"], strict=False)
        return model

    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            obj = dill.load(f)

        if isinstance(obj, nn.Module):
            return obj.cpu()

        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
            return obj["model"].cpu()

        raise ValueError(
            f"{p} does not contain a torch.nn.Module or a dict with key 'model'."
        )

    raise ValueError(f"Unsupported base_model_path extension: {p.suffix}")


# ============================================================================
# Config Builders for Each Precision Type
# Based on MASE src/chop/nn/quantized/functional/linear.py
# ============================================================================
def build_integer_config(trial: optuna.Trial, name: str) -> dict:
    """
    Build config for LinearInteger.
    Required keys: data_in_width, data_in_frac_width, weight_width, weight_frac_width,
                   bias_width, bias_frac_width
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    fw = trial.suggest_categorical(f"{name}_frac_width", FRAC_WIDTH_CHOICES)

    return {
        "data_in_width": w,
        "data_in_frac_width": fw,
        "weight_width": w,
        "weight_frac_width": fw,
        "bias_width": w,
        "bias_frac_width": fw,
    }


def build_minifloat_config(trial: optuna.Trial, name: str) -> dict:
    """
    Build config for LinearMinifloatIEEE and LinearMinifloatDenorm.
    Required keys: data_in_width, data_in_exponent_width, data_in_exponent_bias,
                   weight_width, weight_exponent_width, weight_exponent_bias,
                   bias_width, bias_exponent_width, bias_exponent_bias
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    ew = trial.suggest_categorical(f"{name}_exponent_width", EXPONENT_WIDTH_CHOICES)
    eb = 2 ** (ew - 1) - 1

    return {
        "data_in_width": w,
        "data_in_exponent_width": ew,
        "data_in_exponent_bias": eb,
        "weight_width": w,
        "weight_exponent_width": ew,
        "weight_exponent_bias": eb,
        "bias_width": w,
        "bias_exponent_width": ew,
        "bias_exponent_bias": eb,
    }


def build_log_config(trial: optuna.Trial, name: str) -> dict:
    """
    Build config for LinearLog.
    Required keys: data_in_width, data_in_exponent_bias,
                   weight_width, weight_exponent_bias,
                   bias_width, bias_exponent_bias
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    eb = trial.suggest_categorical(f"{name}_exponent_bias", EXPONENT_BIAS_CHOICES)

    return {
        "data_in_width": w,
        "data_in_exponent_bias": eb,
        "weight_width": w,
        "weight_exponent_bias": eb,
        "bias_width": w,
        "bias_exponent_bias": eb,
    }


def build_block_fp_config(trial: optuna.Trial, name: str, layer: nn.Linear) -> dict:
    """
    Build config for LinearBlockFP.
    Required keys: data_in_width, data_in_exponent_width, data_in_exponent_bias, data_in_block_size,
                   weight_width, weight_exponent_width, weight_exponent_bias, weight_block_size,
                   bias_width, bias_exponent_width, bias_exponent_bias, bias_block_size
    
    Note: block_size must be a list. For 3D activations [batch, seq, hidden] with skip_first_dim=True,
    we need [seq_block, hidden_block]. For 2D weights [out, in], we need [out_block, in_block].
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    ew = trial.suggest_categorical(f"{name}_exponent_width", EXPONENT_WIDTH_CHOICES)
    eb = 2 ** (ew - 1) - 1

    # Block size must be compatible with layer dimensions
    max_block = min(layer.in_features, layer.out_features)
    valid_block_sizes = [bs for bs in BLOCK_SIZE_CHOICES if bs <= max_block]
    if not valid_block_sizes:
        valid_block_sizes = [max_block] if max_block > 0 else [1]

    bs = trial.suggest_categorical(f"{name}_block_size", valid_block_sizes)

    # For 3D input [batch, seq, hidden], block_shape should cover last 2 dims
    # Using [-1, bs] means: don't block seq dim, block hidden dim with bs
    # For 2D weight [out, in], using [bs, bs] or [-1, bs]
    data_in_block_size = [-1, bs]  # Don't block seq_len, block hidden
    weight_block_size = [bs, bs]   # Block both dimensions of weight
    bias_block_size = [bs]         # Bias is 1D

    return {
        "data_in_width": w,
        "data_in_exponent_width": ew,
        "data_in_exponent_bias": eb,
        "data_in_block_size": data_in_block_size,
        "data_in_skip_first_dim": True,
        "weight_width": w,
        "weight_exponent_width": ew,
        "weight_exponent_bias": eb,
        "weight_block_size": weight_block_size,
        "bias_width": w,
        "bias_exponent_width": ew,
        "bias_exponent_bias": eb,
        "bias_block_size": bias_block_size,
    }


def build_block_minifloat_config(trial: optuna.Trial, name: str, layer: nn.Linear) -> dict:
    """
    Build config for LinearBlockMinifloat.
    Required keys: data_in_width, data_in_exponent_width, data_in_exponent_bias_width, data_in_block_size,
                   weight_width, weight_exponent_width, weight_exponent_bias_width, weight_block_size,
                   bias_width, bias_exponent_width, bias_exponent_bias_width, bias_block_size
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    ew = trial.suggest_categorical(f"{name}_exponent_width", EXPONENT_WIDTH_CHOICES)
    ebw = trial.suggest_categorical(f"{name}_exponent_bias_width", EXPONENT_WIDTH_CHOICES)

    # Block size must be compatible with layer dimensions
    max_block = min(layer.in_features, layer.out_features)
    valid_block_sizes = [bs for bs in BLOCK_SIZE_CHOICES if bs <= max_block]
    if not valid_block_sizes:
        valid_block_sizes = [max_block] if max_block > 0 else [1]

    bs = trial.suggest_categorical(f"{name}_block_size", valid_block_sizes)

    data_in_block_size = [-1, bs]
    weight_block_size = [bs, bs]
    bias_block_size = [bs]

    return {
        "data_in_width": w,
        "data_in_exponent_width": ew,
        "data_in_exponent_bias_width": ebw,
        "data_in_block_size": data_in_block_size,
        "data_in_skip_first_dim": True,
        "weight_width": w,
        "weight_exponent_width": ew,
        "weight_exponent_bias_width": ebw,
        "weight_block_size": weight_block_size,
        "bias_width": w,
        "bias_exponent_width": ew,
        "bias_exponent_bias_width": ebw,
        "bias_block_size": bias_block_size,
    }


def build_block_log_config(trial: optuna.Trial, name: str, layer: nn.Linear) -> dict:
    """
    Build config for LinearBlockLog.
    Required keys: data_in_width, data_in_exponent_bias_width, data_in_block_size,
                   weight_width, weight_exponent_bias_width, weight_block_size,
                   bias_width, bias_exponent_bias_width, bias_block_size
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    ebw = trial.suggest_categorical(f"{name}_exponent_bias_width", EXPONENT_WIDTH_CHOICES)

    # Block size must be compatible with layer dimensions
    max_block = min(layer.in_features, layer.out_features)
    valid_block_sizes = [bs for bs in BLOCK_SIZE_CHOICES if bs <= max_block]
    if not valid_block_sizes:
        valid_block_sizes = [max_block] if max_block > 0 else [1]

    bs = trial.suggest_categorical(f"{name}_block_size", valid_block_sizes)

    data_in_block_size = [-1, bs]
    weight_block_size = [bs, bs]
    bias_block_size = [bs]

    return {
        "data_in_width": w,
        "data_in_exponent_bias_width": ebw,
        "data_in_block_size": data_in_block_size,
        "data_in_skip_first_dim": True,
        "weight_width": w,
        "weight_exponent_bias_width": ebw,
        "weight_block_size": weight_block_size,
        "bias_width": w,
        "bias_exponent_bias_width": ebw,
        "bias_block_size": bias_block_size,
    }


def build_binary_config(trial: optuna.Trial, name: str) -> dict:
    """
    Build config for LinearBinary.
    Required keys: weight_stochastic, weight_bipolar
    Note: LinearBinary only quantizes weights, input and bias use passthrough.
    """
    stochastic = trial.suggest_categorical(f"{name}_stochastic", [True, False])
    bipolar = trial.suggest_categorical(f"{name}_bipolar", [True, False])

    return {
        "weight_stochastic": stochastic,
        "weight_bipolar": bipolar,
    }


# Map layer classes to their config builders
# Builders that need layer info return True in second element
CONFIG_BUILDERS = {
    LinearInteger: (build_integer_config, False),
    LinearMinifloatIEEE: (build_minifloat_config, False),
    LinearMinifloatDenorm: (build_minifloat_config, False),
    LinearLog: (build_log_config, False),
    LinearBlockFP: (build_block_fp_config, True),
    LinearBlockMinifloat: (build_block_minifloat_config, True),
    LinearBlockLog: (build_block_log_config, True),
    LinearBinary: (build_binary_config, False),
}


# ============================================================================
# Model Constructor
# ============================================================================
def construct_model(trial: optuna.Trial, base_model: nn.Module) -> tuple[nn.Module, dict]:
    """
    Construct a model with per-layer precision choices sampled by Optuna.

    For each nn.Linear layer, choose from ALL supported precision types:
    [nn.Linear, LinearInteger, LinearMinifloatIEEE, LinearMinifloatDenorm,
     LinearLog, LinearBlockFP, LinearBlockMinifloat, LinearBlockLog, LinearBinary]

    Returns:
        tuple: (model, precision_counts) where precision_counts tracks how many
               layers use each precision type
    """
    trial_model = deepcopy(base_model)
    precision_counts = defaultdict(int)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear):
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                LINEAR_LAYER_CHOICES,
            )

            # Track precision usage
            precision_counts[new_layer_cls.__name__] += 1

            if new_layer_cls == nn.Linear:
                continue

            # Common kwargs
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": (layer.bias is not None),
            }

            # Build precision-specific config
            config_builder, needs_layer = CONFIG_BUILDERS[new_layer_cls]
            if needs_layer:
                kwargs["config"] = config_builder(trial, name, layer)
            else:
                kwargs["config"] = config_builder(trial, name)

            # Create new layer and copy parameters
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None and hasattr(new_layer, "bias") and new_layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias.data)

            deepsetattr(trial_model, name, new_layer)

    return trial_model, dict(precision_counts)


# ============================================================================
# Objective Function
# ============================================================================
def create_objective(dataset, tokenizer, base_model, num_epochs, logger):
    """Create objective function with dataset, tokenizer, base_model, and logger in closure."""

    def objective(trial):
        trial_start_time = datetime.now()

        logger.info(f"Trial {trial.number + 1} started")

        try:
            # Define the model
            model, precision_counts = construct_model(trial, base_model)

            # Count model parameters
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model parameters: {num_params:,}")

            # Log precision distribution
            logger.info(f"  Precision distribution: {precision_counts}")

            trainer = get_trainer(
                model=model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=num_epochs,
            )

            logger.info(f"  Training...")
            trainer.train()

            logger.info(f"  Evaluating...")
            eval_results = trainer.evaluate()
            accuracy = eval_results["eval_accuracy"]

            # Set attributes for later retrieval
            trial.set_user_attr("model", model)
            trial.set_user_attr("num_params", num_params)
            trial.set_user_attr("precision_counts", precision_counts)

            # Calculate trial duration
            trial_duration = datetime.now() - trial_start_time

            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Duration: {trial_duration}")
            logger.info(f"Trial {trial.number + 1} completed")
            logger.info("-" * 50)

            return accuracy

        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Trial {trial.number + 1} failed with error: {e}")
            logger.error(f"Full traceback:\n{full_traceback}")
            raise

    return objective


# ============================================================================
# Optuna Callback for Logging and Incremental Saving
# ============================================================================
class LoggingCallback:
    """Optuna callback to log progress and save results after each trial."""

    def __init__(self, logger, n_trials, output_dir, run_name, start_time):
        self.logger = logger
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.run_name = run_name
        self.start_time = start_time
        self.best_accuracy = 0.0
        self.best_model = None

    def __call__(self, study, trial):
        # Update best accuracy and save best model
        if trial.value and trial.value > self.best_accuracy:
            self.best_accuracy = trial.value
            self.best_model = trial.user_attrs.get("model")
            self.logger.info(f"*** New best accuracy: {self.best_accuracy:.4f} ***")

            # Save best model immediately
            if self.best_model is not None:
                try:
                    best_path = self.output_dir / f"{self.run_name}_best.pt"
                    payload = {
                        "state_dict": {k: v.detach().cpu() for k, v in self.best_model.state_dict().items()},
                        "best_accuracy": self.best_accuracy,
                        "best_params": trial.params,
                        "precision_counts": trial.user_attrs.get("precision_counts", {}),
                        "saved_at": str(datetime.now()),
                    }
                    torch.save(payload, best_path)
                    self.logger.info(f"  Best model saved to: {best_path}")
                except Exception as e:
                    self.logger.warning(f"  Failed to save best model: {e}")

        # Save intermediate results after each trial
        completed_trials = [t for t in study.trials if t.value is not None]
        accuracies = [t.value for t in completed_trials]
        precision_counts_list = [t.user_attrs.get("precision_counts", {}) for t in completed_trials]

        if accuracies:
            try:
                results = {
                    "run_name": self.run_name,
                    "n_trials_completed": len(accuracies),
                    "n_trials_target": self.n_trials,
                    "accuracies": accuracies,
                    "max_acc_per_trial": np.maximum.accumulate(accuracies).tolist(),
                    "best_accuracy": max(accuracies),
                    "best_params": study.best_trial.params if study.best_trial else None,
                    "precision_counts_per_trial": precision_counts_list,
                    "start_time": str(self.start_time),
                    "last_updated": str(datetime.now()),
                    "status": "in_progress",
                }

                results_path = self.output_dir / f"{self.run_name}_results.pkl"
                with open(results_path, "wb") as f:
                    dill.dump(results, f)
            except Exception as e:
                self.logger.warning(f"  Failed to save intermediate results: {e}")

        # Calculate progress
        completed = len(study.trials)
        progress = (completed / self.n_trials) * 100
        elapsed = datetime.now() - self.start_time

        # Estimate remaining time
        if completed > 0:
            avg_time_per_trial = elapsed / completed
            remaining_trials = self.n_trials - completed
            eta = avg_time_per_trial * remaining_trials
        else:
            eta = "Unknown"

        self.logger.info(f"Progress: {completed}/{self.n_trials} ({progress:.1f}%)")
        self.logger.info(f"Elapsed: {elapsed} | ETA: {eta}")
        self.logger.info(f"Best accuracy so far: {self.best_accuracy:.4f}")
        self.logger.info("=" * 50)


# ============================================================================
# Sampler Factory
# ============================================================================
def get_sampler(sampler_type: str):
    """Return the appropriate Optuna sampler based on type string."""

    sampler_type = sampler_type.lower()

    if sampler_type == "random":
        return RandomSampler(seed=SEED), "random"

    elif sampler_type == "tpe":
        return TPESampler(seed=SEED + 1), "tpe"

    else:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            "Choose from: random, tpe"
        )


# ============================================================================
# Plotting
# ============================================================================
def plot_progress(accuracies: list[float], output_path: Path, sampler_name: str) -> None:
    """Plot best accuracy vs trials."""

    best_so_far = np.maximum.accumulate(np.array(accuracies, dtype=float))
    x = np.arange(1, len(best_so_far) + 1)

    plt.figure()
    plt.plot(x, best_so_far, marker="o", markersize=3)
    plt.xlabel("Trial")
    plt.ylabel("Best Accuracy So Far")
    plt.title(f"Mixed Precision Search ({sampler_name.upper()})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_comparison(results_path: Path, output_path: Path) -> None:
    """
    Plot accuracy curves grouped by dominant precision type in each trial.
    
    For each trial, identify the most-used precision type and plot accordingly.
    """
    with open(results_path, "rb") as f:
        results = dill.load(f)

    accuracies = results.get("accuracies", [])
    precision_counts_list = results.get("precision_counts_per_trial", [])

    if not accuracies or not precision_counts_list:
        print("No data to plot")
        return

    # Group trials by dominant precision (most frequently used in that trial)
    precision_accuracies = defaultdict(list)
    precision_trials = defaultdict(list)

    for i, (acc, counts) in enumerate(zip(accuracies, precision_counts_list)):
        if counts:
            # Find dominant precision (excluding nn.Linear for full precision count)
            quantized_counts = {k: v for k, v in counts.items() if k != "Linear"}
            if quantized_counts:
                dominant = max(quantized_counts, key=quantized_counts.get)
            else:
                dominant = "Linear"  # All layers kept as full precision
        else:
            dominant = "Linear"

        precision_accuracies[dominant].append(acc)
        precision_trials[dominant].append(i + 1)

    # Plot
    plt.figure(figsize=(12, 8))

    for precision in sorted(precision_accuracies.keys()):
        trials = precision_trials[precision]
        accs = precision_accuracies[precision]

        # Sort by trial number for proper plotting
        sorted_pairs = sorted(zip(trials, accs))
        trials_sorted = [p[0] for p in sorted_pairs]
        accs_sorted = [p[1] for p in sorted_pairs]

        plt.scatter(trials_sorted, accs_sorted, label=precision, alpha=0.7, s=50)

    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.title("Mixed Precision Search: Accuracy by Dominant Precision Type")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Precision comparison plot saved to: {output_path}")


def plot_precision_best_accuracy(results_path: Path, output_path: Path) -> None:
    """
    Plot best accuracy achieved for each precision type.
    """
    with open(results_path, "rb") as f:
        results = dill.load(f)

    accuracies = results.get("accuracies", [])
    precision_counts_list = results.get("precision_counts_per_trial", [])

    if not accuracies or not precision_counts_list:
        print("No data to plot")
        return

    # Find best accuracy for each dominant precision
    precision_best = defaultdict(float)

    for acc, counts in zip(accuracies, precision_counts_list):
        if counts:
            quantized_counts = {k: v for k, v in counts.items() if k != "Linear"}
            if quantized_counts:
                dominant = max(quantized_counts, key=quantized_counts.get)
            else:
                dominant = "Linear"
        else:
            dominant = "Linear"

        precision_best[dominant] = max(precision_best[dominant], acc)

    # Plot bar chart
    plt.figure(figsize=(12, 6))

    precisions = sorted(precision_best.keys())
    best_accs = [precision_best[p] for p in precisions]

    bars = plt.bar(precisions, best_accs, color="steelblue", edgecolor="black")
    plt.xlabel("Precision Type")
    plt.ylabel("Best Accuracy")
    plt.title("Best Accuracy Achieved by Each Precision Type")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, acc in zip(bars, best_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Best accuracy comparison plot saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Mixed Precision Search with All Supported Precisions")
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        choices=["random", "tpe"],
        help="Sampler type: random or tpe"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "multi_precision_results"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60 * 24,
        help="Timeout in seconds (default: 24 hours)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help="Model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default=TOKENIZER_CHECKPOINT,
        help="Tokenizer checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help="Dataset name"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model from previous NAS stage"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs per trial"
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from existing results"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle plot_only mode
    if args.plot_only:
        sampler, sampler_name = get_sampler(args.sampler)
        run_name = f"multi_precision_{sampler_name}"
        results_path = output_dir / f"{run_name}_results.pkl"

        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return

        plot_precision_comparison(results_path, output_dir / f"{run_name}_precision_scatter.png")
        plot_precision_best_accuracy(results_path, output_dir / f"{run_name}_precision_best.png")
        return

    # Create sampler and get the name for output files
    sampler, sampler_name = get_sampler(args.sampler)
    run_name = f"multi_precision_{sampler_name}"

    # Setup logging
    logger = setup_logging(output_dir, run_name)

    # Log start
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Mixed Precision Search with {args.sampler.upper()} Sampler")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Epochs per trial: {args.epochs}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Base model path: {args.base_model_path}")
    logger.info("=" * 60)

    # Log search space
    logger.info("Search Space:")
    logger.info(f"  Layer choices: [nn.Linear, LinearInteger, LinearMinifloatIEEE,")
    logger.info(f"                  LinearMinifloatDenorm, LinearLog, LinearBlockFP,")
    logger.info(f"                  LinearBlockMinifloat, LinearBlockLog, LinearBinary]")
    logger.info(f"  width_choices: {WIDTH_CHOICES}")
    logger.info(f"  frac_width_choices: {FRAC_WIDTH_CHOICES}")
    logger.info(f"  exponent_width_choices: {EXPONENT_WIDTH_CHOICES}")
    logger.info(f"  exponent_bias_choices: {EXPONENT_BIAS_CHOICES}")
    logger.info(f"  block_size_choices: {BLOCK_SIZE_CHOICES}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=args.dataset,
        checkpoint=args.tokenizer_checkpoint,
        return_tokenizer=True,
    )
    logger.info("Dataset loaded successfully")
    logger.info("=" * 60)

    # Load base model
    logger.info("Loading base model...")
    base_model = load_base_model(args.base_model_path, args.checkpoint)
    logger.info("Base model loaded successfully")
    logger.info("=" * 60)

    logger.info(f"Created {args.sampler.upper()} sampler")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"multi-precision-{sampler_name}",
        sampler=sampler,
    )
    logger.info("Created Optuna study")
    logger.info("=" * 60)

    # Create objective function and callback
    objective = create_objective(dataset, tokenizer, base_model, args.epochs, logger)
    callback = LoggingCallback(logger, args.n_trials, output_dir, run_name, start_time)

    # Run optimization
    logger.info(f"Starting optimization with {args.sampler.upper()} sampler...")
    logger.info("=" * 60)

    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            callbacks=[callback],
            show_progress_bar=False,
        )
    except Exception as e:
        logger.error(f"Optimization stopped due to error: {e}")
        logger.info("Saving results collected so far...")

    # Calculate total duration
    end_time = datetime.now()
    total_duration = end_time - start_time

    # Extract results
    completed_trials = [t for t in study.trials if t.value is not None]
    accuracies = [t.value for t in completed_trials]
    precision_counts_list = [t.user_attrs.get("precision_counts", {}) for t in completed_trials]

    if not accuracies:
        logger.error("No successful trials completed!")
        return

    max_acc_per_trial = np.maximum.accumulate(accuracies)

    # Save final results
    results = {
        "run_name": run_name,
        "sampler": sampler_name,
        "sampler_arg": args.sampler,
        "n_trials_completed": len(accuracies),
        "n_trials_target": args.n_trials,
        "accuracies": accuracies,
        "max_acc_per_trial": max_acc_per_trial.tolist(),
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "precision_counts_per_trial": precision_counts_list,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "total_duration": str(total_duration),
        "status": "completed" if len(accuracies) == args.n_trials else "partial",
    }

    results_path = output_dir / f"{run_name}_results.pkl"
    with open(results_path, "wb") as f:
        dill.dump(results, f)
    logger.info(f"Results saved to: {results_path}")

    # Save best model
    if study.best_trial and "model" in study.best_trial.user_attrs:
        model = study.best_trial.user_attrs["model"]
        best_path = output_dir / f"{run_name}_best.pt"
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_accuracy": study.best_value,
            "best_params": study.best_trial.params,
            "precision_counts": study.best_trial.user_attrs.get("precision_counts", {}),
            "checkpoint": args.checkpoint,
            "saved_at": str(datetime.now()),
        }
        torch.save(payload, best_path)
        logger.info(f"Best model saved to: {best_path}")
    else:
        logger.warning("No best model available to save")

    # Plot progress
    plot_path = output_dir / f"{run_name}_progress.png"
    plot_progress(accuracies, plot_path, sampler_name)
    logger.info(f"Progress plot saved to: {plot_path}")

    # Plot precision comparison
    plot_precision_comparison(results_path, output_dir / f"{run_name}_precision_scatter.png")
    plot_precision_best_accuracy(results_path, output_dir / f"{run_name}_precision_best.png")

    # Log final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FINAL RESULTS - {args.sampler.upper()} Sampler")
    logger.info("=" * 60)
    logger.info(f"Trials completed: {len(accuracies)}/{args.n_trials}")
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best trial: {study.best_trial.number + 1}")
    logger.info(f"Best trial precision distribution: {study.best_trial.user_attrs.get('precision_counts', {})}")
    logger.info("-" * 60)
    logger.info(f"Start time: {start_time}")
    logger.info(f"End time: {end_time}")
    logger.info(f"Total duration: {total_duration}")
    logger.info("=" * 60)

    # Log all trial results
    logger.info("")
    logger.info("All Trial Results:")
    logger.info("-" * 60)
    logger.info(f"{'Trial':<8} {'Accuracy':<12} {'Best So Far':<12}")
    logger.info("-" * 60)
    for i, (acc, best_acc) in enumerate(zip(accuracies, max_acc_per_trial)):
        logger.info(f"{i+1:<8} {acc:<12.4f} {best_acc:<12.4f}")
    logger.info("=" * 60)
    logger.info("Mixed precision search completed!")


if __name__ == "__main__":
    main()