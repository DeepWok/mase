"""
Lab 3 Task 2: Sampler Comparison for Mixed Precision Search

Compare RandomSampler vs TPESampler for per-layer mixed precision type search.
This experiment evaluates how different sampling strategies affect quantization
type selection across BERT layers.

Features:
- Caches results to ~/lab_3_results/task_2_cache.json for resume capability
- Tests 15 trials per sampler (30 total)
- Compares 4 quantization types: FP32, LinearInteger, LinearMinifloatIEEE, LinearLog
- Generates comparison plot: trials vs best accuracy for both samplers
- Saves detailed results to sampler_comparison_results.json
"""

import json
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import RandomSampler, TPESampler

from transformers import AutoModelForSequenceClassification

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

# Experiment config - 100 trials per sampler for production-level comparison
N_TRIALS_PER_SAMPLER = 100

# ============================================================================
# CACHING INFRASTRUCTURE
# ============================================================================

# Results directory
results_dir = Path.home() / "lab_3_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_2_cache.json"

# Load existing cache
if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for samplers: {list(cached_results.keys())}")
else:
    cached_results = {}

def save_cache():
    """Save cached results to disk."""
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)

# ============================================================================
# CONFIG FACTORY FUNCTIONS
# ============================================================================

def get_integer_config(trial, layer_name):
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
        "bias_width": 8,
        "bias_frac_width": 4,
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

# ============================================================================
# SEARCH EXECUTION
# ============================================================================

def run_search(sampler_name, sampler, n_trials):
    """Run NAS with given sampler and track best-so-far accuracy.

    Supports resuming from cached results: if we already have some trials
    cached, only run the additional trials needed to reach n_trials total.

    Args:
        sampler_name: Human-readable name for the sampler
        sampler: Optuna sampler instance
        n_trials: Number of total trials desired

    Returns:
        dict: Results including trial accuracies and best-so-far tracking
    """
    print(f"\n{'='*60}")
    print(f"Running search with {sampler_name}")
    print(f"{'='*60}")

    # Check cache for existing results
    existing_accuracies = []
    existing_best_params = None
    existing_best_acc = 0.0
    if sampler_name in cached_results:
        existing = cached_results[sampler_name]
        existing_accuracies = existing["trial_accuracies"]
        existing_best_params = existing.get("best_params", None)
        existing_best_acc = existing.get("best_accuracy", 0.0)
        existing_count = len(existing_accuracies)
        print(f"  Found {existing_count} cached trials (best: {existing_best_acc:.4f})")

        if existing_count >= n_trials:
            print(f"  Already have {existing_count} >= {n_trials} trials, skipping...")
            return existing
    else:
        existing_count = 0

    remaining = n_trials - existing_count
    print(f"  Running {remaining} additional trials to reach {n_trials} total...")

    # Create study for the new batch of trials
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lab3-task2-{sampler_name}-batch2",
        sampler=sampler,
    )

    # Run the remaining trials
    study.optimize(objective, n_trials=remaining, timeout=60*60*12)

    # Merge old + new trial accuracies
    new_accuracies = [t.value for t in study.trials]
    all_accuracies = existing_accuracies + new_accuracies

    # Recalculate best-so-far across all trials
    best_so_far = []
    current_best = 0.0
    for acc in all_accuracies:
        if acc is not None and acc > current_best:
            current_best = acc
        best_so_far.append(current_best)

    # Determine overall best params
    best_accuracy = max(a for a in all_accuracies if a is not None)
    if existing_best_acc >= study.best_value and existing_best_params is not None:
        best_params = existing_best_params
    else:
        best_params = {k: str(v) for k, v in study.best_params.items()}

    # Store merged results
    result = {
        "trial_accuracies": all_accuracies,
        "best_so_far": best_so_far,
        "best_accuracy": best_accuracy,
        "best_params": best_params,
    }

    # Update cache with merged results
    cached_results[sampler_name] = result
    save_cache()

    print(f"  Best accuracy with {sampler_name}: {best_accuracy:.4f} ({len(all_accuracies)} total trials)")

    return result

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
    print("Dataset loaded successfully!")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    )
    print("Base model loaded successfully!")

    print("\n" + "="*60)
    print("Starting Sampler Comparison: Mixed Precision Search")
    print("="*60)

    # Define samplers to compare
    samplers = {
        "RandomSampler": RandomSampler(seed=42),
        "TPESampler": TPESampler(seed=42),
    }

    # Run experiments
    results = {}
    for name, sampler in samplers.items():
        results[name] = run_search(name, sampler, N_TRIALS_PER_SAMPLER)

    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)

    # ========================================================================
    # PLOT GENERATION
    # ========================================================================

    print("\nGenerating comparison plot...")
    plt.figure(figsize=(10, 6))

    # Colors as specified: RandomSampler=blue, TPESampler=green
    colors = {'RandomSampler': '#3498db', 'TPESampler': '#2ecc71'}
    markers = {'RandomSampler': 's', 'TPESampler': '^'}

    for name, result in results.items():
        trials = list(range(1, len(result["best_so_far"]) + 1))
        plt.plot(
            trials,
            result["best_so_far"],
            marker=markers[name],
            linewidth=2,
            markersize=6,
            color=colors[name],
            label=f'{name} (best: {result["best_accuracy"]:.4f})'
        )

    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
    plt.title('Lab 3 Task 2: Sampler Comparison for Mixed Precision Search',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent / 'lab_3_task_2_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'sampler_comparison_mixed_precision.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================

    print("\n" + "="*60)
    print("SAMPLER COMPARISON ANALYSIS")
    print("="*60)

    # Summary table
    print(f"\n{'Sampler':<20} {'Best Accuracy':<15} {'Trials to 95%':<15}")
    print("-"*50)

    for sampler_name, result in results.items():
        # Calculate trials to reach 95% of best
        target = result["best_accuracy"] * 0.95
        trials_to_95 = None
        for i, acc in enumerate(result["best_so_far"]):
            if acc >= target:
                trials_to_95 = i + 1
                break

        trials_str = str(trials_to_95) if trials_to_95 else "N/A"
        print(f"{sampler_name:<20} {result['best_accuracy']:<15.4f} {trials_str:<15}")

    # Winner analysis
    print("\n" + "-"*50)
    best_sampler = max(results.items(), key=lambda x: x[1]["best_accuracy"])
    print(f"Best sampler: {best_sampler[0]} with {best_sampler[1]['best_accuracy']:.4f}")

    # Efficiency analysis
    random_best = results.get("RandomSampler", {}).get("best_accuracy", 0)
    tpe_best = results.get("TPESampler", {}).get("best_accuracy", 0)

    if tpe_best > random_best:
        improvement = (tpe_best - random_best) / random_best * 100
        print(f"TPESampler improvement over RandomSampler: {improvement:.2f}%")
    elif random_best > tpe_best:
        improvement = (random_best - tpe_best) / tpe_best * 100
        print(f"RandomSampler performed better by: {improvement:.2f}%")
    else:
        print("Both samplers achieved the same best accuracy")

    print("="*60)

    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================

    final_results = {
        "n_trials_per_sampler": N_TRIALS_PER_SAMPLER,
        "total_trials": N_TRIALS_PER_SAMPLER * len(samplers),
        "quantization_types": [cls.__name__ for cls in QUANTIZATION_TYPES],
        "samplers": results,
    }

    results_file = output_dir / 'sampler_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to '{results_file}'")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    # Convergence analysis
    for name, result in results.items():
        # Check early vs late improvements
        early_improvement = result["best_so_far"][4] - result["best_so_far"][0]  # First 5 trials
        late_improvement = result["best_so_far"][-1] - result["best_so_far"][-5]  # Last 5 trials

        print(f"\n{name}:")
        print(f"  Early improvement (trials 1-5): {early_improvement:.4f}")
        print(f"  Late improvement (trials {len(result['best_so_far'])-4}-{len(result['best_so_far'])}): {late_improvement:.4f}")

        if early_improvement > late_improvement:
            print(f"  -> Converged quickly, most gains in early trials")
        else:
            print(f"  -> Continued improving, late gains significant")

    print("\n" + "="*60)
    print("Experiment complete! Check the plot and JSON files for detailed results.")
