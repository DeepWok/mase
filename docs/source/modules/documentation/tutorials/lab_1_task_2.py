"""
Lab 1 Task 2: Pruning Sparsity Analysis

Take the best model from Task 1 and vary pruning sparsity from 0.1 to 0.9.
Compare Random vs L1-Norm pruning methods to evaluate the effect of
different pruning strategies on IMDb sentiment analysis accuracy.

PREREQUISITE: Run lab_1_task_1.py first!
This script requires the best QAT model saved by Task 1 at:
  ~/task_1_best_model
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer

# Configuration
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# Use the best model from Task 1
# This model is automatically saved by lab_1_task_1.py
# It will be the bit-width that achieved the highest QAT accuracy
base_checkpoint_preferred = f"{Path.home()}/task_1_best_model"
base_checkpoint_fallback = f"{Path.home()}/tutorial_3_qat"

# Verify the checkpoint exists
checkpoint_path_pt = Path(base_checkpoint_preferred + ".pt")
checkpoint_path_mz = Path(base_checkpoint_preferred + ".mz")
fallback_pt = Path(base_checkpoint_fallback + ".pt")

if checkpoint_path_pt.exists() and checkpoint_path_mz.exists():
    base_checkpoint = base_checkpoint_preferred
    print("=" * 60)
    print("✓ Using best model from Task 1")
    print("=" * 60)
elif fallback_pt.exists():
    base_checkpoint = base_checkpoint_fallback
    print("=" * 60)
    print("⚠ WARNING: Task 1 checkpoint not found!")
    print("Using fallback: Tutorial 3 QAT model (8-bit)")
    print("=" * 60)
    print("For better results, run lab_1_task_1.py first to find")
    print("the optimal quantization bit-width for your model.")
    print("=" * 60)
else:
    print("=" * 60)
    print("ERROR: No valid checkpoint found!")
    print("=" * 60)
    print(f"Tried:")
    print(f"  1. {checkpoint_path_pt}")
    print(f"  2. {fallback_pt}")
    print("\nPlease either:")
    print("  - Run lab_1_task_1.py to generate the best model, OR")
    print("  - Run Tutorial 3 to generate tutorial_3_qat checkpoint")
    print("=" * 60)
    exit(1)

print(f"Base checkpoint: {base_checkpoint}")
print("=" * 60)

# Sparsity levels to test (0.1 = 10% pruned, 0.9 = 90% pruned)
sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Pruning methods to compare
pruning_methods = ["random", "l1-norm"]

# Storage for results
results = {method: {"before_finetune": [], "after_finetune": []}
           for method in pruning_methods}

# Load dataset once (reuse for all experiments)
print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

print("=" * 60)
print("Starting Pruning Experiments")
print(f"Base model: {base_checkpoint}")
print("=" * 60)

for method in pruning_methods:
    print(f"\n{'#'*60}")
    print(f"PRUNING METHOD: {method.upper()}")
    print(f"{'#'*60}")

    for sparsity in sparsity_levels:
        print(f"\n{'='*60}")
        print(f"Testing sparsity={sparsity:.1f} with {method}")
        print(f"{'='*60}")

        # 1. Load fresh model for each experiment
        print(f"  Loading model from checkpoint...")
        mg = MaseGraph.from_checkpoint(base_checkpoint)

        # 2. Create pruning config for this sparsity and method
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

        # 3. Apply pruning pass
        print(f"  Applying pruning transform ({method}, sparsity={sparsity:.1f})...")
        mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

        # 4. Fine-tune the pruned model (skip pre-evaluation for speed)
        print(f"  Fine-tuning pruned model (1 epoch)...")
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )

        trainer.train()

        # 5. Evaluate after fine-tuning
        after_results = trainer.evaluate()
        after_acc = after_results['eval_accuracy']
        results[method]["after_finetune"].append(after_acc)
        print(f"  ✓ After fine-tuning: {after_acc:.4f}")

        # Store dummy "before" value for compatibility (we're not using it)
        results[method]["before_finetune"].append(0.5)  # placeholder

print("\n" + "=" * 60)
print("Experiments Complete!")
print("=" * 60)

# 7. Create visualization
print("\nGenerating plot...")
plt.figure(figsize=(10, 6))

# Main comparison plot: Random vs L1-Norm after fine-tuning
for method in pruning_methods:
    color = '#e74c3c' if method == 'random' else '#3498db'
    marker = 'o' if method == 'random' else 's'
    plt.plot(sparsity_levels, results[method]["after_finetune"],
             marker=marker, linewidth=2, markersize=8,
             label=f'{method.upper()}',
             color=color, linestyle='-')

# Add baseline (no pruning)
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
print(f"✓ Plot saved as '{output_file}'")
plt.show()

# 8. Print summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Sparsity':<12} {'Random':<12} {'L1-Norm':<12} {'Difference':<12}")
print("-" * 60)
for i, sparsity in enumerate(sparsity_levels):
    random_acc = results["random"]["after_finetune"][i]
    l1_acc = results["l1-norm"]["after_finetune"][i]
    diff = l1_acc - random_acc
    print(f"{sparsity:<12.1f} {random_acc:<12.4f} {l1_acc:<12.4f} {diff:+.4f}")
print("=" * 60)
print(f"Baseline (no pruning): {baseline_acc:.4f}")
print("=" * 60)

# 9. Save results for later analysis
import json

save_results = {
    'sparsity_levels': sparsity_levels,
    'baseline_accuracy': baseline_acc,
    'random': results["random"]["after_finetune"],
    'l1_norm': results["l1-norm"]["after_finetune"],
}

results_file = output_dir / 'pruning_results.json'
with open(results_file, 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"\n✓ Results saved to '{results_file}'")

# 10. Key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

# Find best sparsity for each method
for method in pruning_methods:
    best_idx = np.argmax(results[method]["after_finetune"])
    best_sparsity = sparsity_levels[best_idx]
    best_acc = results[method]["after_finetune"][best_idx]
    print(f"{method.upper()}: Best accuracy {best_acc:.4f} at sparsity={best_sparsity:.1f}")

# Compare methods
random_after = np.array(results["random"]["after_finetune"])
l1_after = np.array(results["l1-norm"]["after_finetune"])
diff = l1_after - random_after
avg_diff = np.mean(diff)

if avg_diff > 0:
    print(f"\nL1-Norm outperforms Random by {avg_diff:.4f} on average")
else:
    print(f"\nRandom outperforms L1-Norm by {-avg_diff:.4f} on average")

# Find maximum sparsity maintaining >95% baseline accuracy
threshold = baseline_acc * 0.95
print(f"\nMaximum sparsity maintaining >{threshold:.4f} accuracy:")
for method in pruning_methods:
    max_sparsity = 0
    for i, acc in enumerate(results[method]["after_finetune"]):
        if acc >= threshold:
            max_sparsity = sparsity_levels[i]
    if max_sparsity > 0:
        print(f"  {method.upper()}: {max_sparsity:.1f} ({max_sparsity*100:.0f}% parameters pruned)")
    else:
        print(f"  {method.upper()}: Cannot maintain threshold")

print("=" * 60)
