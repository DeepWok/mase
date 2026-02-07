"""
Lab 2 Task 1: NAS Sampler Comparison

Compare three Optuna samplers (GridSampler, RandomSampler, TPESampler) by tracking
maximum achieved accuracy up to each trial point. This experiment evaluates how
different sampling strategies affect Neural Architecture Search efficiency.

Features:
- Caches results to ~/lab_2_results/task_1_cache.json for resume capability
- Tests 10 trials per sampler (increase to 50-100 for production)
- Generates comparison plot: trials vs best accuracy
- Saves detailed results to sampler_comparison_results.json
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and dataset config (from Tutorial 5)
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# Search space for hyperparameters
# GridSampler requires enumerable space: 3 * 4 * 5 * 5 = 300 combinations
# We'll use a subset for Grid, full for Random/TPE
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
}

# GridSampler needs explicit search space dict with indices
grid_search_space = {
    "num_layers": list(range(len(search_space["num_layers"]))),
    "num_heads": list(range(len(search_space["num_heads"]))),
    "hidden_size": list(range(len(search_space["hidden_size"]))),
    "intermediate_size": list(range(len(search_space["intermediate_size"]))),
}

# Experiment config
N_TRIALS = 100  # Production-level trial count

# ============================================================================
# CACHING INFRASTRUCTURE
# ============================================================================

# Results directory
results_dir = Path.home() / "lab_2_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_1_cache.json"

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
# DATASET LOADING
# ============================================================================

print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
print("Dataset loaded successfully!")

# ============================================================================
# MODEL CONSTRUCTOR
# ============================================================================

def construct_model(trial):
    """Construct a BERT model with sampled hyperparameters.

    Args:
        trial: Optuna trial object for sampling

    Returns:
        AutoModelForSequenceClassification with sampled architecture
    """
    config = AutoConfig.from_pretrained(checkpoint)

    # Sample hyperparameters using trial.suggest_int
    # This is compatible with all samplers (Grid, Random, TPE)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    # Build model from config
    model = AutoModelForSequenceClassification.from_config(config)
    return model

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """Train and evaluate a sampled model.

    Args:
        trial: Optuna trial object

    Returns:
        float: Evaluation accuracy on IMDb test set
    """
    # Construct model with sampled hyperparameters
    model = construct_model(trial)

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

    return eval_results["eval_accuracy"]

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
        study_name=f"lab2-{sampler_name}-continued",
        sampler=sampler,
    )

    # Run the remaining trials
    study.optimize(objective, n_trials=remaining, timeout=60*60*24)

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
        best_params = {k: v for k, v in study.best_params.items()}

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

print("\n" + "="*60)
print("Starting NAS Sampler Comparison")
print("="*60)

# Define samplers
samplers = {
    "GridSampler": GridSampler(grid_search_space),
    "RandomSampler": RandomSampler(),
    "TPESampler": TPESampler(),
}

# Run experiments
results = {}
for name, sampler in samplers.items():
    results[name] = run_search(name, sampler, N_TRIALS)

print("\n" + "="*60)
print("All experiments complete!")
print("="*60)

# ============================================================================
# PLOT GENERATION
# ============================================================================

print("\nGenerating plot...")
plt.figure(figsize=(10, 6))

colors = {'GridSampler': '#e74c3c', 'RandomSampler': '#3498db', 'TPESampler': '#2ecc71'}
markers = {'GridSampler': 'o', 'RandomSampler': 's', 'TPESampler': '^'}

for name, result in results.items():
    trials = list(range(1, len(result["best_so_far"]) + 1))
    plt.plot(trials, result["best_so_far"],
             marker=markers[name], linewidth=2, markersize=6,
             label=f'{name} (best: {result["best_accuracy"]:.4f})',
             color=colors[name])

plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
plt.title('NAS Sampler Comparison: Best Accuracy vs Trial Number',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save plot
output_dir = Path(__file__).parent / 'lab_2_task_1_results'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'sampler_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_file}'")
plt.show()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

# Print summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Sampler':<20} {'Best Accuracy':<15} {'Trial':<10}")
print("-"*60)
for name, result in results.items():
    best_trial = result["best_so_far"].index(max(result["best_so_far"])) + 1
    print(f"{name:<20} {result['best_accuracy']:<15.4f} {best_trial:<10}")
print("="*60)

# Save final results
final_results = {
    "n_trials": N_TRIALS,
    "samplers": results,
}
results_file = output_dir / 'sampler_comparison_results.json'
with open(results_file, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to '{results_file}'")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

# Find best sampler
best_sampler = max(results.items(), key=lambda x: x[1]["best_accuracy"])
print(f"Best sampler: {best_sampler[0]} with accuracy {best_sampler[1]['best_accuracy']:.4f}")

# Convergence analysis
for name, result in results.items():
    # Check how many trials to reach 95% of best accuracy
    target = result["best_accuracy"] * 0.95
    trials_to_95 = None
    for i, acc in enumerate(result["best_so_far"]):
        if acc >= target:
            trials_to_95 = i + 1
            break
    if trials_to_95:
        print(f"{name}: Reached 95% of best in {trials_to_95} trials")
    else:
        print(f"{name}: Did not reach 95% of best")

print("="*60)
print("\nExperiment complete! Check the plot and JSON files for detailed results.")
