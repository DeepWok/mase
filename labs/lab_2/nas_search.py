#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) with Mase and Optuna

Usage:
    python nas_search.py --sampler random --n_trials 20
    python nas_search.py --sampler tpe --n_trials 20
    python nas_search.py --sampler grid-linear --n_trials 20
    python nas_search.py --sampler grid-identity --n_trials 20

Run all four in parallel on HPC:
    python nas_search.py --sampler random --n_trials 20 &
    python nas_search.py --sampler tpe --n_trials 20 &
    python nas_search.py --sampler grid-linear --n_trials 20 &
    python nas_search.py --sampler grid-identity --n_trials 20 &
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import dill
import torch.nn as nn

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity


# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"
SEED = 42

# Search space (same as notebook)
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


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(output_dir: Path, sampler_name: str) -> logging.Logger:
    """Set up logging to both file and console."""
    
    logger = logging.getLogger(f"nas_{sampler_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    log_file = output_dir / f"nas_log_{sampler_name}.txt"
    file_handler = logging.FileHandler(log_file, mode='w')
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
# def construct_model(trial):
#     config = AutoConfig.from_pretrained(CHECKPOINT)

#     for param in [
#         "num_layers",
#         "num_heads",
#         "hidden_size",
#         "intermediate_size",
#     ]:
#         chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
#         setattr(config, param, search_space[param][chosen_idx])

#     trial_model = AutoModelForSequenceClassification.from_config(config)

#     for name, layer in trial_model.named_modules():
#         if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
#             new_layer_cls = trial.suggest_categorical(
#                 f"{name}_type",
#                 search_space["linear_layer_choices"],
#             )

#             if new_layer_cls == nn.Linear:
#                 continue
#             elif new_layer_cls == Identity:
#                 new_layer = Identity()
#                 deepsetattr(trial_model, name, new_layer)
#             else:
#                 raise ValueError(f"Unknown layer type: {new_layer_cls}")

#     return trial_model

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
# Objective Function
# ============================================================================
def create_objective(dataset, tokenizer, logger):
    """Create objective function with dataset, tokenizer, and logger in closure."""
    
    def objective(trial):
        trial_start_time = datetime.now()
        
        logger.info(f"Trial {trial.number + 1} started")
        
        try:
            # Define the model
            model = construct_model(trial)
            
            # Log the sampled hyperparameters
            params = {
                "num_layers": trial.params["num_layers"],
                "num_heads": trial.params["num_heads"],
                "hidden_size": trial.params["hidden_size"],
                "intermediate_size": trial.params["intermediate_size"],
            }

            logger.info(f"  Hyperparameters: {params}")
            
            # Count model parameters
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model parameters: {num_params:,}")

            trainer = get_trainer(
                model=model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=3,
            )

            logger.info(f"  Training...")
            trainer.train()
            
            logger.info(f"  Evaluating...")
            eval_results = trainer.evaluate()
            accuracy = eval_results["eval_accuracy"]

            # Set the model as an attribute so we can fetch it later
            trial.set_user_attr("model", model)
            trial.set_user_attr("num_params", num_params)
            
            # Calculate trial duration
            trial_duration = datetime.now() - trial_start_time
            
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Duration: {trial_duration}")
            logger.info(f"Trial {trial.number + 1} completed")
            logger.info("-" * 50)

            return accuracy
        
        except ValueError as e:
            # Handle cache collision errors from HuggingFace evaluate
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
            
            # Save best model immediately
            if self.best_model is not None:
                try:
                    model_path = self.output_dir / f"nas_best_model_{self.sampler_name}.pkl"
                    with open(model_path, "wb") as f:
                        dill.dump(self.best_model.cpu(), f)
                    self.logger.info(f"  Best model saved to: {model_path}")
                except Exception as e:
                    self.logger.warning(f"  Failed to save best model: {e}")
        
        # Save intermediate results after each trial
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
# def get_sampler(sampler_type: str):
#     """
#     Return the appropriate Optuna sampler based on type string.
    
#     Returns:
#         tuple: (sampler, sampler_name_for_files)
#     """
    
#     sampler_type = sampler_type.lower()
    
#     if sampler_type == "random":
#         return RandomSampler(seed=SEED), "random"
    
#     elif sampler_type == "tpe":
#         return TPESampler(seed=SEED), "tpe"
    
#     elif sampler_type == "grid-linear":
#         # GridSampler with nn.Linear only (keep all linear layers)
#         search_space["linear_layer_choices"] = [nn.Linear]
        
#         grid_search_space = {
#             "num_layers": list(range(len(search_space["num_layers"]))),
#             "num_heads": list(range(len(search_space["num_heads"]))),
#             "hidden_size": list(range(len(search_space["hidden_size"]))),
#             "intermediate_size": list(range(len(search_space["intermediate_size"]))),
#         }
#         return GridSampler(search_space=grid_search_space), "grid_linear"
    
#     elif sampler_type == "grid-identity":
#         # GridSampler with Identity only (replace all matching linear layers)
#         search_space["linear_layer_choices"] = [Identity]
        
#         grid_search_space = {
#             "num_layers": list(range(len(search_space["num_layers"]))),
#             "num_heads": list(range(len(search_space["num_heads"]))),
#             "hidden_size": list(range(len(search_space["hidden_size"]))),
#             "intermediate_size": list(range(len(search_space["intermediate_size"]))),
#         }
#         return GridSampler(search_space=grid_search_space), "grid_identity"
    
#     # Keep backwards compatibility with just "grid" (defaults to linear)
#     elif sampler_type == "grid":
#         search_space["linear_layer_choices"] = [nn.Linear]
        
#         grid_search_space = {
#             "num_layers": list(range(len(search_space["num_layers"]))),
#             "num_heads": list(range(len(search_space["num_heads"]))),
#             "hidden_size": list(range(len(search_space["hidden_size"]))),
#             "intermediate_size": list(range(len(search_space["intermediate_size"]))),
#         }
#         return GridSampler(search_space=grid_search_space), "grid"
    
#     else:
#         raise ValueError(
#             f"Unknown sampler type: {sampler_type}. "
#             "Choose from: random, tpe, grid, grid-linear, grid-identity"
#         )

def get_sampler(sampler_type: str):
    """Return the appropriate Optuna sampler based on type string."""
    sampler_type = sampler_type.lower()

    if sampler_type == "random":
        return RandomSampler(seed=SEED), "random"

    elif sampler_type == "tpe":
        return TPESampler(seed=SEED+1), "tpe"

    elif sampler_type == "grid":
        # Fix B: grid over REAL values
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
    parser = argparse.ArgumentParser(description="NAS with Optuna - Sampler Comparison")
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        # choices=["random", "tpe", "grid", "grid-linear", "grid-identity"],
        choices=["random", "tpe", "grid"],
        help="Sampler type: random, tpe, or grid"
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
        default=str(Path.home()),
        help="Directory to save results (default: home directory)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60 * 24,
        help="Timeout in seconds (default: 24 hours)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sampler and get the name for output files
    sampler, sampler_name = get_sampler(args.sampler)

    # Setup logging (use sampler_name for file naming)
    logger = setup_logging(output_dir, sampler_name)
    
    # Log start
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"NAS Search with {args.sampler.upper()} Sampler")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file prefix: {sampler_name}")
    logger.info(f"Checkpoint: {CHECKPOINT}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info("=" * 60)
    
    # Log search space
    logger.info("Search Space:")
    for key, values in search_space.items():
        if key != "linear_layer_choices":
            logger.info(f"  {key}: {values}")
        else:
            # Show what layer choices are actually being used
            layer_names = []
            for cls in values:
                if cls == nn.Linear:
                    layer_names.append("nn.Linear")
                elif cls == Identity:
                    layer_names.append("Identity")
            logger.info(f"  {key}: [{', '.join(layer_names)}]")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=DATASET_NAME,
        checkpoint=TOKENIZER_CHECKPOINT,
        return_tokenizer=True,
    )
    logger.info("Dataset loaded successfully")
    logger.info("=" * 60)

    logger.info(f"Created {args.sampler.upper()} sampler")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-nas-{sampler_name}",
        sampler=sampler,
    )
    logger.info("Created Optuna study")
    logger.info("=" * 60)

    # Create objective function and callback
    objective = create_objective(dataset, tokenizer, logger)
    callback = LoggingCallback(logger, args.n_trials, output_dir, sampler_name, start_time)

    # Run optimization
    logger.info(f"Starting optimization with {args.sampler.upper()} sampler...")
    logger.info("=" * 60)
    
    # Suppress Optuna's default logging to avoid duplicate messages
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

    # Extract results (only from successful trials)
    completed_trials = [t for t in study.trials if t.value is not None]
    accuracies = [t.value for t in completed_trials]
    
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
        "start_time": str(start_time),
        "end_time": str(end_time),
        "total_duration": str(total_duration),
        "status": "completed" if len(accuracies) == args.n_trials else "partial",
    }

    results_path = output_dir / f"nas_results_{sampler_name}.pkl"
    with open(results_path, "wb") as f:
        dill.dump(results, f)
    logger.info(f"Results saved to: {results_path}")

    # Save best model
    if study.best_trial and "model" in study.best_trial.user_attrs:
        model = study.best_trial.user_attrs["model"].cpu()
        model_path = output_dir / f"nas_best_model_{sampler_name}.pkl"
        with open(model_path, "wb") as f:
            dill.dump(model, f)
        logger.info(f"Best model saved to: {model_path}")
    else:
        logger.warning("No best model available to save")

    # Log final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FINAL RESULTS - {args.sampler.upper()} Sampler")
    logger.info("=" * 60)
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
    
    # Log all trial results
    logger.info("")
    logger.info("All Trial Results:")
    logger.info("-" * 60)
    logger.info(f"{'Trial':<8} {'Accuracy':<12} {'Best So Far':<12}")
    logger.info("-" * 60)
    for i, (acc, best_acc) in enumerate(zip(accuracies, max_acc_per_trial)):
        logger.info(f"{i+1:<8} {acc:<12.4f} {best_acc:<12.4f}")
    logger.info("=" * 60)
    logger.info("NAS search completed!")


if __name__ == "__main__":
    main()