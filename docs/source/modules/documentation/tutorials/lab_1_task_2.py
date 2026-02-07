"""
Lab 1 Task 2: Pruning Sparsity Analysis

Take the best model from Task 1 and vary pruning sparsity from 0.1 to 0.9.
Compare Random vs L1-Norm pruning methods to evaluate the effect of
different pruning strategies on IMDb sentiment analysis accuracy.

PREREQUISITE: Run lab_1_task_1.py first!
Requires the best QAT model saved by Task 1 at ~/task_1_best_model
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

base_checkpoint_preferred = f"{Path.home()}/task_1_best_model"
base_checkpoint_fallback = f"{Path.home()}/tutorial_3_qat"

checkpoint_path_pt = Path(base_checkpoint_preferred + ".pt")
checkpoint_path_mz = Path(base_checkpoint_preferred + ".mz")
fallback_pt = Path(base_checkpoint_fallback + ".pt")

if checkpoint_path_pt.exists() and checkpoint_path_mz.exists():
    base_checkpoint = base_checkpoint_preferred
    print("Using best model from Task 1")
elif fallback_pt.exists():
    base_checkpoint = base_checkpoint_fallback
    print("WARNING: Task 1 checkpoint not found, using fallback Tutorial 3 QAT model")
else:
    print("ERROR: No valid checkpoint found!")
    print(f"Tried: {checkpoint_path_pt}, {fallback_pt}")
    print("Run lab_1_task_1.py first or Tutorial 3 to generate a checkpoint")
    exit(1)

print(f"Base checkpoint: {base_checkpoint}")

sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pruning_methods = ["random", "l1-norm"]

results = {method: {"before_finetune": [], "after_finetune": []}
           for method in pruning_methods}

print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

for method in pruning_methods:
    print(f"\nPruning method: {method.upper()}")

    for sparsity in sparsity_levels:
        print(f"  Loading model for sparsity={sparsity:.1f}...")
        mg = MaseGraph.from_checkpoint(base_checkpoint)

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

        mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )
        trainer.train()

        after_results = trainer.evaluate()
        after_acc = after_results['eval_accuracy']
        results[method]["after_finetune"].append(after_acc)
        results[method]["before_finetune"].append(0.5)  # placeholder
        print(f"  {method} sparsity={sparsity:.1f}: {after_acc:.4f}")

# Plot
plt.figure(figsize=(10, 6))

for method in pruning_methods:
    color = '#e74c3c' if method == 'random' else '#3498db'
    marker = 'o' if method == 'random' else 's'
    plt.plot(sparsity_levels, results[method]["after_finetune"],
             marker=marker, linewidth=2, markersize=8,
             label=f'{method.upper()}',
             color=color, linestyle='-')

print("Evaluating baseline (no pruning)...")
baseline_mg = MaseGraph.from_checkpoint(base_checkpoint)
baseline_trainer = get_trainer(
    model=baseline_mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
baseline_results = baseline_trainer.evaluate()
baseline_acc = baseline_results['eval_accuracy']
plt.axhline(y=baseline_acc, color='gray', linestyle=':',
            linewidth=1.5, label=f'Baseline (no pruning): {baseline_acc:.4f}')

plt.xlabel('Sparsity Level', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy on IMDb Dataset', fontsize=12, fontweight='bold')
plt.title('Pruning Method Comparison: Random vs L1-Norm\n(After 1 Epoch Fine-tuning)',
          fontsize=13, fontweight='bold', pad=15)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(sparsity_levels)
plt.ylim([0.5, max(baseline_acc + 0.05, 0.85)])

plt.tight_layout()
output_dir = Path(__file__).parent / 'lab_1_task_2_results'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'pruning_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_file}'")
plt.show()

# Summary
print(f"\n{'Sparsity':<12} {'Random':<12} {'L1-Norm':<12} {'Difference':<12}")
print("-" * 48)
for i, sparsity in enumerate(sparsity_levels):
    random_acc = results["random"]["after_finetune"][i]
    l1_acc = results["l1-norm"]["after_finetune"][i]
    diff = l1_acc - random_acc
    print(f"{sparsity:<12.1f} {random_acc:<12.4f} {l1_acc:<12.4f} {diff:+.4f}")
print(f"Baseline (no pruning): {baseline_acc:.4f}")

# Save results
save_results = {
    'sparsity_levels': sparsity_levels,
    'baseline_accuracy': baseline_acc,
    'random': results["random"]["after_finetune"],
    'l1_norm': results["l1-norm"]["after_finetune"],
}
results_file = output_dir / 'pruning_results.json'
with open(results_file, 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"Results saved to '{results_file}'")

for method in pruning_methods:
    best_idx = np.argmax(results[method]["after_finetune"])
    best_sparsity = sparsity_levels[best_idx]
    best_acc = results[method]["after_finetune"][best_idx]
    print(f"{method.upper()}: Best accuracy {best_acc:.4f} at sparsity={best_sparsity:.1f}")

random_after = np.array(results["random"]["after_finetune"])
l1_after = np.array(results["l1-norm"]["after_finetune"])
avg_diff = np.mean(l1_after - random_after)
print(f"L1-Norm outperforms Random by {avg_diff:.4f} on average")

threshold = baseline_acc * 0.95
print(f"\nMax sparsity maintaining >{threshold:.4f} accuracy:")
for method in pruning_methods:
    max_sparsity = 0
    for i, acc in enumerate(results[method]["after_finetune"]):
        if acc >= threshold:
            max_sparsity = sparsity_levels[i]
    if max_sparsity > 0:
        print(f"  {method.upper()}: {max_sparsity:.1f} ({max_sparsity*100:.0f}% pruned)")
    else:
        print(f"  {method.upper()}: Cannot maintain threshold")
