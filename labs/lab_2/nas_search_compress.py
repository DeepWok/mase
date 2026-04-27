#!/usr/bin/env python3
"""
Compression-Aware Neural Architecture Search (NAS) with Mase and Optuna

This script runs NAS with compression (quantization + pruning) applied during search.
Two modes are available:
- Without post-compression training: Train (N epochs) → Compress → Evaluate
- With post-compression training: Train (N epochs) → Compress → Train (M epochs) → Evaluate

Output naming format: nas_results_compress_{sampler}_{no_post|with_post}_{epochs_before}_{epochs_after}.pkl
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import dill
import copy  
import torch  
import torch.nn as nn

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity
from chop.pipelines import CompressionPipeline
from chop import MaseGraph


# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"
SEED = 42

# Search space
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [
        nn.Linear,
        Identity,
    ],
}

# Compression configs
QUANTIZATION_CONFIG = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

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


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(output_dir: Path, sampler_name: str) -> logging.Logger:
    """Set up logging to both file and console."""

    logger = logging.getLogger(f"nas_{sampler_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log_file = output_dir / f"nas_log_{sampler_name}.txt"
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
# Model Constructor
# ============================================================================
def construct_model(trial):
    """
    Construct a model with hyperparameters sampled by Optuna.

    Uses suggest_categorical with real values (not indices) for cleaner params.
    GridSampler skips per-layer choices since it can't handle dynamic parameter names.
    """
    config = AutoConfig.from_pretrained(CHECKPOINT)

    # Sample REAL values and set the correct HuggingFace config fields
    config.num_hidden_layers = trial.suggest_categorical("num_layers", search_space["num_layers"])
    config.num_attention_heads = trial.suggest_categorical("num_heads", search_space["num_heads"])
    config.hidden_size = trial.suggest_categorical("hidden_size", search_space["hidden_size"])
    config.intermediate_size = trial.suggest_categorical("intermediate_size", search_space["intermediate_size"])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    # GridSampler can't handle dynamic parameter names like f"{name}_type"
    if isinstance(trial.study.sampler, GridSampler):
        return trial_model

    # For Random/TPE: allow per-layer choices
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )
            if new_layer_cls == Identity:
                deepsetattr(trial_model, name, Identity())

    return trial_model


# ============================================================================
# Compression Function
# ============================================================================
def compress_model(model, logger):
    """
    Apply quantization and pruning to the model using Mase CompressionPipeline.
    """
    logger.info("    Applying compression (quantization + pruning)...")

    # keep a reference to original model object
    orig_model = model

    try:
        # Remember original device from the original model
        original_device = next(orig_model.parameters()).device

        # Move a CPU reference for MaseGraph
        model_cpu = orig_model.cpu()

        mg = MaseGraph(
            model_cpu,
            hf_input_names=["input_ids", "attention_mask", "labels"],  # optionally add token_type_ids
        )
        pipe = CompressionPipeline()

        #  Deep-copy configs so pipeline can't mutate globals across trials
        quant_cfg = copy.deepcopy(QUANTIZATION_CONFIG)
        prune_cfg = copy.deepcopy(PRUNING_CONFIG)

        mg, _ = pipe(
            mg,
            pass_args={
                "quantize_transform_pass": quant_cfg,
                "prune_transform_pass": prune_cfg,
            },
        )

        # Move compressed model back to original device (GPU)
        compressed_model = mg.model.to(original_device)

        logger.info(f"    Compression applied successfully (device: {original_device})")
        return compressed_model

    except Exception as e:
        logger.warning(f"    Compression failed: {e}")
        logger.warning("    Returning original model")
        try:
            return orig_model.to(original_device)
        except Exception:
            return orig_model


# ============================================================================
# Objective Function
# ============================================================================
def create_objective(dataset, tokenizer, logger, epochs_before, epochs_after, post_training=False):
    """
    Create objective function for compression-aware NAS.
    """

    def objective(trial):
        trial_start_time = datetime.now()

        logger.info(f"Trial {trial.number + 1} started")

        try:
            # Step 1: Construct the model
            model = construct_model(trial)

            params = {
                "num_layers": trial.params["num_layers"],
                "num_heads": trial.params["num_heads"],
                "hidden_size": trial.params["hidden_size"],
                "intermediate_size": trial.params["intermediate_size"],
            }
            logger.info(f"  Hyperparameters: {params}")

            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model parameters: {num_params:,}")

            # Step 2: Pre-compression training
            logger.info(f"  [BEFORE COMPRESSION] Training for {epochs_before} epoch(s)...")
            trainer = get_trainer(
                model=model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=epochs_before,
            )
            trainer.train()

            # Step 3: Apply compression
            logger.info("  [COMPRESSION] Applying quantization and pruning...")
            compressed_model = compress_model(model, logger)

            num_params_compressed = sum(p.numel() for p in compressed_model.parameters())
            num_nonzero = sum((p != 0).sum().item() for p in compressed_model.parameters())
            logger.info(f"  Compressed model: {num_params_compressed:,} params (non-zero: {num_nonzero:,})")

            # Step 4: Post-compression training (optional)
            if post_training:
                logger.info(f"  [AFTER COMPRESSION] Training for {epochs_after} epoch(s)...")
                trainer_post = get_trainer(
                    model=compressed_model,
                    tokenized_dataset=dataset,
                    tokenizer=tokenizer,
                    evaluate_metric="accuracy",
                    num_train_epochs=epochs_after,
                )
                trainer_post.train()

                logger.info("  [EVALUATE] After post-compression training...")
                eval_results = trainer_post.evaluate()
            else:
                logger.info("  [EVALUATE] Directly after compression (no post-training)...")
                trainer_eval = get_trainer(
                    model=compressed_model,
                    tokenized_dataset=dataset,
                    tokenizer=tokenizer,
                    evaluate_metric="accuracy",
                    num_train_epochs=1,  # not used for eval; only for trainer init
                )
                eval_results = trainer_eval.evaluate()

            accuracy = eval_results["eval_accuracy"]

            # Save model and metadata (kept as-is)
            trial.set_user_attr("model", compressed_model)
            trial.set_user_attr("num_params", num_params)
            trial.set_user_attr("num_params_compressed", num_params_compressed)
            trial.set_user_attr("num_nonzero", num_nonzero)

            trial_duration = datetime.now() - trial_start_time

            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Duration: {trial_duration}")
            logger.info(f"Trial {trial.number + 1} completed")
            logger.info("-" * 50)

            return accuracy

        except ValueError as e:
            if "cache file" in str(e):
                logger.warning(f"Trial {trial.number + 1} failed due to cache collision: {e}")
                logger.warning("Returning 0.0 for this trial and continuing...")
                return 0.0
            raise

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

            if self.best_model is not None:
                try:
                    # save state_dict instead of pickling the whole module
                    model_path = self.output_dir / f"nas_best_model_{self.sampler_name}.pt"
                    payload = {
                        "state_dict": {k: v.detach().cpu() for k, v in self.best_model.state_dict().items()},
                        "best_accuracy": float(self.best_accuracy),
                        "best_params": study.best_trial.params if study.best_trial else None,
                        "sampler_name": self.sampler_name,
                        "saved_at": str(datetime.now()),
                        "checkpoint": CHECKPOINT,
                    }
                    torch.save(payload, model_path)
                    self.logger.info(f"  Best model state_dict saved to: {model_path}")
                except Exception as e:
                    self.logger.warning(f"  Failed to save best model state_dict: {e}")

        # Save intermediate results after each trial (kept as-is)
        completed_trials = [t for t in study.trials if t.value is not None]
        accuracies = [t.value for t in completed_trials]

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
                    "start_time": str(self.start_time),
                    "last_updated": str(datetime.now()),
                    "status": "in_progress",
                }

                results_path = self.output_dir / f"nas_results_{self.sampler_name}.pkl"
                with open(results_path, "wb") as f:
                    dill.dump(results, f)
            except Exception as e:
                self.logger.warning(f"  Failed to save intermediate results: {e}")

        # Progress logging (kept as-is)
        completed = len(study.trials)
        progress = (completed / self.n_trials) * 100
        elapsed = datetime.now() - self.start_time

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

    elif sampler_type == "grid":
        grid_search_space = {
            "num_layers": search_space["num_layers"],
            "num_heads": search_space["num_heads"],
            "hidden_size": search_space["hidden_size"],
            "intermediate_size": search_space["intermediate_size"],
        }
        return GridSampler(search_space=grid_search_space), "grid"

    else:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            "Choose from: random, tpe, grid"
        )


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Compression-Aware NAS with Optuna")
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        choices=["random", "tpe", "grid"],
        help="Sampler type: random, tpe, or grid",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=str(Path.home()))
    parser.add_argument("--timeout", type=int, default=60 * 60 * 24)
    parser.add_argument("--epochs_before", type=int, default=2, choices=range(1, 11))
    parser.add_argument("--epochs_after", type=int, default=1, choices=range(1, 11))

    post_training_group = parser.add_mutually_exclusive_group(required=True)
    post_training_group.add_argument("--post_training", action="store_true", dest="post_training")
    post_training_group.add_argument("--no_post_training", action="store_false", dest="post_training")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler, base_sampler_name = get_sampler(args.sampler)

    if args.post_training:
        sampler_name = f"compress_{base_sampler_name}_with_post_{args.epochs_before}_{args.epochs_after}"
    else:
        sampler_name = f"compress_{base_sampler_name}_no_post_{args.epochs_before}_{args.epochs_after}"

    logger = setup_logging(output_dir, sampler_name)

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Compression-Aware NAS Search")
    logger.info("=" * 60)
    logger.info(f"Sampler: {args.sampler.upper()}")
    logger.info(f"Post-compression training: {args.post_training}")
    logger.info(f"Epochs BEFORE compression: {args.epochs_before}")
    logger.info(f"Epochs AFTER compression: {args.epochs_after if args.post_training else 'N/A'}")
    logger.info(f"Start time: {start_time}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file prefix: {sampler_name}")
    logger.info(f"Checkpoint: {CHECKPOINT}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info("=" * 60)

    logger.info("Loading dataset...")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=DATASET_NAME,
        checkpoint=TOKENIZER_CHECKPOINT,
        return_tokenizer=True,
    )
    logger.info("Dataset loaded successfully")
    logger.info("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-nas-{sampler_name}",
        sampler=sampler,
    )

    objective = create_objective(
        dataset,
        tokenizer,
        logger,
        epochs_before=args.epochs_before,
        epochs_after=args.epochs_after,
        post_training=args.post_training,
    )
    callback = LoggingCallback(logger, args.n_trials, output_dir, sampler_name, start_time)

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

    end_time = datetime.now()
    total_duration = end_time - start_time

    completed_trials = [t for t in study.trials if t.value is not None]
    accuracies = [t.value for t in completed_trials]

    if not accuracies:
        logger.error("No successful trials completed!")
        return

    max_acc_per_trial = np.maximum.accumulate(accuracies)

    results = {
        "sampler": sampler_name,
        "sampler_base": base_sampler_name,
        "post_training": args.post_training,
        "epochs_before": args.epochs_before,
        "epochs_after": args.epochs_after if args.post_training else None,
        "compression_aware": True,
        "n_trials_completed": len(accuracies),
        "n_trials_target": args.n_trials,
        "accuracies": accuracies,
        "max_acc_per_trial": max_acc_per_trial.tolist(),
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "total_duration": str(total_duration),
        "status": "completed" if len(accuracies) == args.n_trials else "partial",
        "quantization_config": QUANTIZATION_CONFIG,
        "pruning_config": PRUNING_CONFIG,
    }

    results_path = output_dir / f"nas_results_{sampler_name}.pkl"
    with open(results_path, "wb") as f:
        dill.dump(results, f)
    logger.info(f"Results saved to: {results_path}")

    #final best-model save (also as state_dict)
    if study.best_trial and "model" in study.best_trial.user_attrs:
        best_model = study.best_trial.user_attrs["model"]
        model_path = output_dir / f"nas_best_model_{sampler_name}_final.pt"
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in best_model.state_dict().items()},
            "best_accuracy": float(study.best_value),
            "best_params": study.best_params,
            "sampler_name": sampler_name,
            "saved_at": str(datetime.now()),
            "checkpoint": CHECKPOINT,
        }
        torch.save(payload, model_path)
        logger.info(f"Best model state_dict saved to: {model_path}")
    else:
        logger.warning("No best model available to save")

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS - Compression-Aware NAS")
    logger.info("=" * 60)
    logger.info(f"Sampler: {args.sampler.upper()}")
    logger.info(f"Post-compression training: {args.post_training}")
    logger.info(f"Epochs before compression: {args.epochs_before}")
    logger.info(f"Epochs after compression: {args.epochs_after if args.post_training else 'N/A'}")
    logger.info(f"Trials completed: {len(accuracies)}/{args.n_trials}")
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best trial: {study.best_trial.number + 1}")
    logger.info("Best hyperparameters:")
    for param, val in study.best_params.items():
        if param in {"num_layers", "num_heads", "hidden_size", "intermediate_size"}:
            logger.info(f"  {param}: {val}")
    logger.info("-" * 60)
    logger.info(f"Start time: {start_time}")
    logger.info(f"End time: {end_time}")
    logger.info(f"Total duration: {total_duration}")
    logger.info("=" * 60)

    logger.info("")
    logger.info("All Trial Results:")
    logger.info("-" * 60)
    logger.info(f"{'Trial':<8} {'Accuracy':<12} {'Best So Far':<12}")
    logger.info("-" * 60)
    for i, (acc, best_acc) in enumerate(zip(accuracies, max_acc_per_trial)):
        logger.info(f"{i+1:<8} {acc:<12.4f} {best_acc:<12.4f}")
    logger.info("=" * 60)
    logger.info("Compression-aware NAS search completed!")


if __name__ == "__main__":
    main()
