#!/usr/bin/env python3
"""
Tutorial 6 - Task 1: Mixed Precision Search with Mase and Optuna

Goal:
- Replace nn.Linear layers with LinearInteger or keep full precision
- For each layer chosen as LinearInteger, choose:
    width in [8, 16, 32]
    frac_width in [2, 4, 8]
- Expose these per-layer choices as Optuna hyperparameters
- Run Optuna search and plot: trial index vs best accuracy so far

Usage:
    python mixed_precision_search.py --sampler random --n_trials 20
    python mixed_precision_search.py --sampler tpe --n_trials 20

Run both in parallel on HPC:
    python mixed_precision_search.py --sampler random --n_trials 20 &
    python mixed_precision_search.py --sampler tpe --n_trials 20 &
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from copy import deepcopy
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
from chop.nn.quantized.modules.linear import LinearInteger


# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"
SEED = 42

# Search space for Task 1
search_space = {
    "linear_layer_choices": [nn.Linear, LinearInteger],
    "width_choices": [8, 16, 32],
    "frac_width_choices": [2, 4, 8],
}


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(output_dir: Path, sampler_name: str) -> logging.Logger:
    """Set up logging to both file and console."""

    logger = logging.getLogger(f"mixed_precision_{sampler_name}")
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

    log_file = output_dir / f"mixed_precision_log_{sampler_name}.txt"
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

    Supports:
      - .pt payload saved as {"state_dict", "best_params", "checkpoint"}
      - .pkl containing either a full model object or a dict with "model" key
      - fallback: instantiate from checkpoint if no path is provided
    """
    if base_model_path is None:
        return AutoModelForSequenceClassification.from_pretrained(checkpoint)

    p = Path(base_model_path)
    if not p.exists():
        raise FileNotFoundError(f"Base model path does not exist: {p}")

    # Preferred: .pt payload with state_dict
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
# Model Constructor
# ============================================================================
def construct_model(trial: optuna.Trial, base_model: nn.Module) -> nn.Module:
    """
    Construct a model with per-layer quantization choices sampled by Optuna.

    For each nn.Linear layer:
    - Choose [nn.Linear, LinearInteger]
    - If LinearInteger: choose width and frac_width
    """
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear):
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == nn.Linear:
                continue

            # Common kwargs
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": (layer.bias is not None),
            }

            if new_layer_cls == LinearInteger:
                # Per-layer integer format hyperparameters
                w = trial.suggest_categorical(f"{name}_width", search_space["width_choices"])
                fw = trial.suggest_categorical(f"{name}_frac_width", search_space["frac_width_choices"])

                kwargs["config"] = {
                    "data_in_width": w,
                    "data_in_frac_width": fw,
                    "weight_width": w,
                    "weight_frac_width": fw,
                    "bias_width": w,
                    "bias_frac_width": fw,
                }

            # Create new layer and copy parameters
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None and hasattr(new_layer, "bias") and new_layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias.data)

            deepsetattr(trial_model, name, new_layer)

    return trial_model


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
            model = construct_model(trial, base_model)

            # Count model parameters
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model parameters: {num_params:,}")

            # Count quantized layers and collect their configs
            num_quantized = 0
            width_counts = {8: 0, 16: 0, 32: 0}
            frac_width_counts = {2: 0, 4: 0, 8: 0}
            
            for name, m in model.named_modules():
                if isinstance(m, LinearInteger):
                    num_quantized += 1
                    # Extract width from config
                    if hasattr(m, 'config') and m.config:
                        w = m.config.get('weight_width', None)
                        fw = m.config.get('weight_frac_width', None)
                        if w in width_counts:
                            width_counts[w] += 1
                        if fw in frac_width_counts:
                            frac_width_counts[fw] += 1
            
            logger.info(f"  Quantized layers: {num_quantized}")
            logger.info(f"  Width distribution: {width_counts}")
            logger.info(f"  Frac_width distribution: {frac_width_counts}")

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
            trial.set_user_attr("num_quantized", num_quantized)
            trial.set_user_attr("width_counts", width_counts)
            trial.set_user_attr("frac_width_counts", frac_width_counts)

            # Calculate trial duration
            trial_duration = datetime.now() - trial_start_time

            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Duration: {trial_duration}")
            logger.info(f"Trial {trial.number + 1} completed")
            logger.info("-" * 50)

            return accuracy

        except Exception as e:
            logger.error(f"Trial {trial.number + 1} failed with error: {e}")
            raise

    return objective


# ============================================================================
# Optuna Callback for Logging and Incremental Saving
# ============================================================================
class LoggingCallback:
    """Optuna callback to log progress and save results after each trial."""

    def __init__(self, logger, n_trials, output_dir, sampler_name, start_time):
        self.logger = logger
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.sampler_name = sampler_name
        self.best_accuracy = 0.0
        self.best_model = None
        self.start_time = start_time

    def __call__(self, study, trial):
        # Update best accuracy and save best model
        if trial.value and trial.value > self.best_accuracy:
            self.best_accuracy = trial.value
            self.best_model = trial.user_attrs.get("model")
            self.logger.info(f"*** New best accuracy: {self.best_accuracy:.4f} ***")

            # Save best model immediately
            if self.best_model is not None:
                try:
                    best_path = self.output_dir / f"mixed_precision_best_{self.sampler_name}.pt"
                    payload = {
                        "state_dict": {k: v.detach().cpu() for k, v in self.best_model.state_dict().items()},
                        "best_accuracy": self.best_accuracy,
                        "best_params": trial.params,
                        "saved_at": str(datetime.now()),
                    }
                    torch.save(payload, best_path)
                    self.logger.info(f"  Best model saved to: {best_path}")
                except Exception as e:
                    self.logger.warning(f"  Failed to save best model: {e}")

        # Save intermediate results after each trial
        completed_trials = [t for t in study.trials if t.value is not None]
        accuracies = [t.value for t in completed_trials]
        width_counts_list = [t.user_attrs.get("width_counts", {}) for t in completed_trials]
        frac_width_counts_list = [t.user_attrs.get("frac_width_counts", {}) for t in completed_trials]
        num_quantized_list = [t.user_attrs.get("num_quantized", 0) for t in completed_trials]

        if accuracies:
            try:
                results = {
                    "sampler": self.sampler_name,
                    "n_trials_completed": len(accuracies),
                    "n_trials_target": self.n_trials,
                    "accuracies": accuracies,
                    "max_acc_per_trial": np.maximum.accumulate(accuracies).tolist(),
                    "best_accuracy": max(accuracies),
                    "best_params": study.best_trial.params if study.best_trial else None,
                    "width_counts_per_trial": width_counts_list,
                    "frac_width_counts_per_trial": frac_width_counts_list,
                    "num_quantized_per_trial": num_quantized_list,
                    "start_time": str(self.start_time),
                    "last_updated": str(datetime.now()),
                    "status": "in_progress",
                }

                results_path = self.output_dir / f"mixed_precision_results_{self.sampler_name}.pkl"
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


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Mixed Precision Search with Optuna")
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
        default=str(Path.home() / "mixed_precision_results"),
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sampler and get the name for output files
    sampler, sampler_name = get_sampler(args.sampler)

    # Setup logging
    logger = setup_logging(output_dir, sampler_name)

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
    logger.info(f"  linear_layer_choices: [nn.Linear, LinearInteger]")
    logger.info(f"  width_choices: {search_space['width_choices']}")
    logger.info(f"  frac_width_choices: {search_space['frac_width_choices']}")
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
        study_name=f"mixed-precision-{sampler_name}",
        sampler=sampler,
    )
    logger.info("Created Optuna study")
    logger.info("=" * 60)

    # Create objective function and callback
    objective = create_objective(dataset, tokenizer, base_model, args.epochs, logger)
    callback = LoggingCallback(logger, args.n_trials, output_dir, sampler_name, start_time)

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
    width_counts_list = [t.user_attrs.get("width_counts", {}) for t in completed_trials]
    frac_width_counts_list = [t.user_attrs.get("frac_width_counts", {}) for t in completed_trials]
    num_quantized_list = [t.user_attrs.get("num_quantized", 0) for t in completed_trials]

    if not accuracies:
        logger.error("No successful trials completed!")
        return

    max_acc_per_trial = np.maximum.accumulate(accuracies)

    # Save final results
    results = {
        "sampler": sampler_name,
        "sampler_arg": args.sampler,
        "n_trials_completed": len(accuracies),
        "n_trials_target": args.n_trials,
        "accuracies": accuracies,
        "max_acc_per_trial": max_acc_per_trial.tolist(),
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "width_counts_per_trial": width_counts_list,
        "frac_width_counts_per_trial": frac_width_counts_list,
        "num_quantized_per_trial": num_quantized_list,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "total_duration": str(total_duration),
        "status": "completed" if len(accuracies) == args.n_trials else "partial",
        "width_choices": search_space["width_choices"],
        "frac_width_choices": search_space["frac_width_choices"],
    }

    results_path = output_dir / f"mixed_precision_results_{sampler_name}.pkl"
    with open(results_path, "wb") as f:
        dill.dump(results, f)
    logger.info(f"Results saved to: {results_path}")

    # Save best model
    if study.best_trial and "model" in study.best_trial.user_attrs:
        model = study.best_trial.user_attrs["model"]
        best_path = output_dir / f"mixed_precision_best_{sampler_name}.pt"
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_accuracy": study.best_value,
            "best_params": study.best_trial.params,
            "checkpoint": args.checkpoint,
            "saved_at": str(datetime.now()),
        }
        torch.save(payload, best_path)
        logger.info(f"Best model saved to: {best_path}")
    else:
        logger.warning("No best model available to save")

    # Plot progress
    plot_path = output_dir / f"mixed_precision_progress_{sampler_name}.png"
    plot_progress(accuracies, plot_path, sampler_name)
    logger.info(f"Progress plot saved to: {plot_path}")

    # Log final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FINAL RESULTS - {args.sampler.upper()} Sampler")
    logger.info("=" * 60)
    logger.info(f"Trials completed: {len(accuracies)}/{args.n_trials}")
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best trial: {study.best_trial.number + 1}")
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