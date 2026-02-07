"""
Lab 3 Task 1: Per-Layer Mixed Precision Search

Implements per-layer quantization type search using Optuna TPESampler. Each Linear layer
independently samples from 4 quantization types (FP32, LinearInteger, LinearMinifloatIEEE,
LinearLog) with type-specific hyperparameters.

Features:
- Caches results to ~/lab_3_results/task_1_cache.json for resume capability
- Tests 20 trials with TPESampler (increase for production)
- Generates plot: trials vs best accuracy
- Saves detailed results to mixed_precision_results.json
"""

# Standard library
import json
from pathlib import Path
from copy import deepcopy

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import TPESampler

# Transformers
from transformers import AutoModelForSequenceClassification

# Mase imports
import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatIEEE,
    LinearLog,
)
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and dataset config (consistent with Labs 1-2)
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# Quantization type choices for per-layer search
QUANTIZATION_TYPES = [
    torch.nn.Linear,      # FP32 baseline (no quantization)
    LinearInteger,        # Fixed-point integer quantization
    LinearMinifloatIEEE,  # IEEE-style minifloat
    LinearLog,            # Logarithmic quantization
]

# Per-layer width/frac_width search space
WIDTH_CHOICES = [8, 16, 32]
FRAC_WIDTH_CHOICES = [2, 4, 8]

# Minifloat-specific: exponent width and bias
EXPONENT_WIDTH_CHOICES = [3, 4, 5]
EXPONENT_BIAS_CHOICES = [7, 15]

# Experiment config
N_TRIALS = 20  # Increase to 50-100 for production

# Results directory
results_dir = Path.home() / "lab_3_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_1_cache.json"

# ============================================================================
# CACHING INFRASTRUCTURE
# ============================================================================

# Load existing cache
if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Loaded {len(cached_results.get('trials', []))} cached trials")
else:
    cached_results = {"trials": [], "best_so_far": [], "best_config": None}

def save_cache():
    """Save cached results to disk."""
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)

# ============================================================================
# CONFIG FACTORY FUNCTIONS
# ============================================================================

def get_integer_config(trial, layer_name):
    """Generate config for LinearInteger layer.

    Required params: data_in_width, data_in_frac_width,
                     weight_width, weight_frac_width,
                     bias_width, bias_frac_width

    Constraint: frac_width < width
    """
    # Sample widths
    data_in_width = trial.suggest_categorical(
        f"{layer_name}_data_in_width", WIDTH_CHOICES
    )
    weight_width = trial.suggest_categorical(
        f"{layer_name}_weight_width", WIDTH_CHOICES
    )

    # Always sample from the full list (Optuna requires static choice sets)
    data_in_frac_width = trial.suggest_categorical(
        f"{layer_name}_data_in_frac_width", FRAC_WIDTH_CHOICES
    )
    weight_frac_width = trial.suggest_categorical(
        f"{layer_name}_weight_frac_width", FRAC_WIDTH_CHOICES
    )

    # Clamp: ensure frac_width < width
    if data_in_frac_width >= data_in_width:
        data_in_frac_width = data_in_width // 2
    if weight_frac_width >= weight_width:
        weight_frac_width = weight_width // 2

    return {
        "data_in_width": data_in_width,
        "data_in_frac_width": data_in_frac_width,
        "weight_width": weight_width,
        "weight_frac_width": weight_frac_width,
        "bias_width": 8,       # Fixed for stability
        "bias_frac_width": 4,  # Fixed for stability
    }

def get_minifloat_ieee_config(trial, layer_name):
    """Generate config for LinearMinifloatIEEE layer.

    Required params: data_in_width, data_in_exponent_width, data_in_exponent_bias,
                     weight_width, weight_exponent_width, weight_exponent_bias,
                     bias_width, bias_exponent_width, bias_exponent_bias
    """
    data_in_width = trial.suggest_categorical(
        f"{layer_name}_mf_data_in_width", WIDTH_CHOICES
    )
    data_in_exp_width = trial.suggest_categorical(
        f"{layer_name}_mf_data_in_exp_width", EXPONENT_WIDTH_CHOICES
    )

    weight_width = trial.suggest_categorical(
        f"{layer_name}_mf_weight_width", WIDTH_CHOICES
    )
    weight_exp_width = trial.suggest_categorical(
        f"{layer_name}_mf_weight_exp_width", EXPONENT_WIDTH_CHOICES
    )

    return {
        "data_in_width": data_in_width,
        "data_in_exponent_width": data_in_exp_width,
        "data_in_exponent_bias": 7,  # Standard IEEE bias
        "weight_width": weight_width,
        "weight_exponent_width": weight_exp_width,
        "weight_exponent_bias": 7,
        "bias_width": 16,
        "bias_exponent_width": 5,
        "bias_exponent_bias": 15,
    }

def get_log_config(trial, layer_name):
    """Generate config for LinearLog layer.

    Required params: data_in_width, data_in_exponent_bias,
                     weight_width, weight_exponent_bias,
                     bias_width, bias_exponent_bias
    """
    data_in_width = trial.suggest_categorical(
        f"{layer_name}_log_data_in_width", WIDTH_CHOICES
    )
    weight_width = trial.suggest_categorical(
        f"{layer_name}_log_weight_width", WIDTH_CHOICES
    )

    return {
        "data_in_width": data_in_width,
        "data_in_exponent_bias": 7,
        "weight_width": weight_width,
        "weight_exponent_bias": 7,
        "bias_width": 16,
        "bias_exponent_bias": 15,
    }

def get_config_for_type(layer_cls, trial, layer_name):
    """Dispatch to appropriate config factory based on layer type."""
    if layer_cls == torch.nn.Linear:
        return None  # No config needed for FP32
    elif layer_cls == LinearInteger:
        return get_integer_config(trial, layer_name)
    elif layer_cls == LinearMinifloatIEEE:
        return get_minifloat_ieee_config(trial, layer_name)
    elif layer_cls == LinearLog:
        return get_log_config(trial, layer_name)
    else:
        raise ValueError(f"Unknown layer type: {layer_cls}")

# ============================================================================
# MODEL CONSTRUCTOR
# ============================================================================

def construct_model(trial, base_model):
    """Construct model with per-layer quantization sampled by Optuna.

    For each Linear layer:
    1. Sample quantization type from QUANTIZATION_TYPES
    2. Get type-specific config via factory
    3. Create new layer with config
    4. Copy weights from original
    5. Replace in model via deepsetattr
    """
    trial_model = deepcopy(base_model)

    # Track layer configurations for logging
    layer_configs = {}

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            # Sample quantization type for this layer
            layer_type_idx = trial.suggest_int(
                f"{name}_type_idx", 0, len(QUANTIZATION_TYPES) - 1
            )
            new_layer_cls = QUANTIZATION_TYPES[layer_type_idx]

            # Track for logging
            layer_configs[name] = {
                "type": new_layer_cls.__name__,
                "type_idx": layer_type_idx,
            }

            # Skip if FP32 (no change needed)
            if new_layer_cls == torch.nn.Linear:
                continue

            # Get type-specific config
            config = get_config_for_type(new_layer_cls, trial, name)
            layer_configs[name]["config"] = config

            # Create new layer
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
                "config": config,
            }
            new_layer = new_layer_cls(**kwargs)

            # Copy weights
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()

            # Replace layer in model
            deepsetattr(trial_model, name, new_layer)

    # Store config for later analysis
    trial.set_user_attr("layer_configs", layer_configs)

    return trial_model

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """Train and evaluate a model with sampled per-layer quantization.

    Returns: eval_accuracy on IMDb test set
    """
    try:
        # Construct quantized model
        model = construct_model(trial, base_model)

        # Create trainer
        trainer = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()

        accuracy = eval_results["eval_accuracy"]

        # Log trial result
        print(f"Trial {trial.number}: accuracy={accuracy:.4f}")

        return accuracy
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a low value to indicate failure
        return 0.0

# ============================================================================
# SEARCH EXECUTION
# ============================================================================

def run_search(n_trials):
    """Execute mixed precision search with TPESampler."""
    print("\n" + "="*60)
    print("Starting Per-Layer Mixed Precision Search")
    print(f"Trials: {n_trials}, Sampler: TPESampler")
    print("="*60)

    # Check if we have cached results
    if cached_results["trials"] and len(cached_results["trials"]) >= n_trials:
        print("Using cached results...")
        return cached_results

    # Create study with TPESampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name="lab3-mixed-precision",
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=60*60*24)

    # Calculate best-so-far at each trial
    best_so_far = []
    for trial_num in range(len(study.trials)):
        accuracies = [t.value for t in study.trials[:trial_num+1] if t.value is not None]
        best_so_far.append(max(accuracies) if accuracies else 0)

    # Prepare results
    results = {
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": {k: str(v) for k, v in t.params.items()},
                "layer_configs": t.user_attrs.get("layer_configs", {}),
            }
            for t in study.trials
        ],
        "best_so_far": best_so_far,
        "best_accuracy": study.best_value,
        "best_params": {k: str(v) for k, v in study.best_params.items()},
        "best_trial_number": study.best_trial.number,
    }

    # Cache results
    cached_results.update(results)
    save_cache()

    print(f"\nBest accuracy: {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial.number}")

    return results

# ============================================================================
# PLOT GENERATION
# ============================================================================

def generate_plot(results):
    """Generate plot: trials vs maximum achieved accuracy."""
    plt.figure(figsize=(10, 6))

    trials = list(range(1, len(results["best_so_far"]) + 1))

    plt.plot(
        trials,
        results["best_so_far"],
        marker='o',
        linewidth=2,
        markersize=6,
        color='#2ecc71',
        label=f'TPESampler (best: {results["best_accuracy"]:.4f})'
    )

    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
    plt.title('Lab 3 Task 1: Per-Layer Mixed Precision Search\n(TPESampler)',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent / 'lab_3_task_1_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'mixed_precision_search.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    plt.show()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def print_summary(results):
    """Print summary of search results."""
    print("\n" + "="*60)
    print("SEARCH SUMMARY")
    print("="*60)
    print(f"Total trials: {len(results['trials'])}")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Best trial: {results['best_trial_number']}")
    print("\nBest configuration (layer types):")

    # Extract best config
    best_trial = results['trials'][results['best_trial_number']]
    for layer_name, config in best_trial.get('layer_configs', {}).items():
        print(f"  {layer_name}: {config['type']}")

    print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    )

    # Run search
    results = run_search(N_TRIALS)

    # Generate plot
    generate_plot(results)

    # Print summary
    print_summary(results)

    # Save final results
    output_dir = Path(__file__).parent / 'lab_3_task_1_results'
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / 'mixed_precision_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{results_file}'")
