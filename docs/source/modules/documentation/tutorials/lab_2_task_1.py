"""
Lab 2 Task 1: NAS Sampler Comparison

Compare three Optuna samplers (GridSampler, RandomSampler, TPESampler) by tracking
maximum achieved accuracy up to each trial point. This experiment evaluates how
different sampling strategies affect Neural Architecture Search efficiency.

Features:
- Caches results to ~/lab_2_results/task_1_cache.json for resume capability
- Tests 100 trials per sampler
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

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
}

grid_search_space = {
    "num_layers": list(range(len(search_space["num_layers"]))),
    "num_heads": list(range(len(search_space["num_heads"]))),
    "hidden_size": list(range(len(search_space["hidden_size"]))),
    "intermediate_size": list(range(len(search_space["intermediate_size"]))),
}

N_TRIALS = 100

results_dir = Path.home() / "lab_2_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_1_cache.json"

if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for samplers: {list(cached_results.keys())}")
else:
    cached_results = {}

def save_cache():
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)

print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])
    return AutoModelForSequenceClassification.from_config(config)

def objective(trial):
    model = construct_model(trial)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]

def run_search(sampler_name, sampler, n_trials):
    existing_accuracies = []
    existing_best_params = None
    existing_best_acc = 0.0
    if sampler_name in cached_results:
        existing = cached_results[sampler_name]
        existing_accuracies = existing["trial_accuracies"]
        existing_best_params = existing.get("best_params", None)
        existing_best_acc = existing.get("best_accuracy", 0.0)
        existing_count = len(existing_accuracies)
        print(f"  {sampler_name}: {existing_count} cached trials (best: {existing_best_acc:.4f})")

        if existing_count >= n_trials:
            print(f"  Already have {existing_count} >= {n_trials} trials, skipping...")
            return existing
    else:
        existing_count = 0

    remaining = n_trials - existing_count
    print(f"  Running {remaining} additional trials for {sampler_name}...")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"lab2-{sampler_name}-continued",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=remaining, timeout=60*60*24)

    new_accuracies = [t.value for t in study.trials]
    all_accuracies = existing_accuracies + new_accuracies

    best_so_far = []
    current_best = 0.0
    for acc in all_accuracies:
        if acc is not None and acc > current_best:
            current_best = acc
        best_so_far.append(current_best)

    best_accuracy = max(a for a in all_accuracies if a is not None)
    if existing_best_acc >= study.best_value and existing_best_params is not None:
        best_params = existing_best_params
    else:
        best_params = {k: v for k, v in study.best_params.items()}

    result = {
        "trial_accuracies": all_accuracies,
        "best_so_far": best_so_far,
        "best_accuracy": best_accuracy,
        "best_params": best_params,
    }

    cached_results[sampler_name] = result
    save_cache()

    print(f"  Best accuracy with {sampler_name}: {best_accuracy:.4f} ({len(all_accuracies)} total trials)")
    return result

# Run experiments
samplers = {
    "GridSampler": GridSampler(grid_search_space),
    "RandomSampler": RandomSampler(),
    "TPESampler": TPESampler(),
}

results = {}
for name, sampler in samplers.items():
    results[name] = run_search(name, sampler, N_TRIALS)

# Plot
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

output_dir = Path(__file__).parent / 'lab_2_task_1_results'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'sampler_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_file}'")
plt.show()

# Summary
print(f"\n{'Sampler':<20} {'Best Accuracy':<15} {'Trial':<10}")
print("-" * 45)
for name, result in results.items():
    best_trial = result["best_so_far"].index(max(result["best_so_far"])) + 1
    print(f"{name:<20} {result['best_accuracy']:<15.4f} {best_trial:<10}")

final_results = {
    "n_trials": N_TRIALS,
    "samplers": results,
}
results_file = output_dir / 'sampler_comparison_results.json'
with open(results_file, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to '{results_file}'")

best_sampler = max(results.items(), key=lambda x: x[1]["best_accuracy"])
print(f"Best sampler: {best_sampler[0]} with accuracy {best_sampler[1]['best_accuracy']:.4f}")

for name, result in results.items():
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
