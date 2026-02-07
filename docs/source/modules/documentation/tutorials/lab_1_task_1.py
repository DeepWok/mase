"""
Lab 1 Task 1: Quantization Precision Analysis

Two experiments:
1. Balanced sweep: Total bit widths from 4 to 32 (frac = width//2), PTQ vs QAT.
2. Imbalanced sweep: For total widths {8, 12, 16}, vary frac_width independently
   (e.g. Q4.8 = 4 integer bits, 8 fractional bits). Generates a heatmap.

Features:
- Automatically saves each trained model to ~/task_1_models/
- Caches results to skip already-trained models (enables resume on interrupt)
- Saves the best QAT model to ~/task_1_best_model for use in Task 2

To force re-training: Delete ~/task_1_models/ directory
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

# Bit widths to test
bit_widths = [4, 6, 8, 12, 16, 24, 32]

# Storage for results
ptq_accuracies = []
qat_accuracies = []

# Directory for saving individual models
models_dir = Path.home() / "task_1_models"
models_dir.mkdir(exist_ok=True)
results_cache_file = models_dir / "results_cache.json"

# Load cached results if they exist
import json
if results_cache_file.exists():
    with open(results_cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"✓ Found cached results for {len(cached_results)} models")
else:
    cached_results = {}

# Track progress
models_from_cache = 0
models_newly_trained = 0

# Load dataset once (reuse for all experiments)
print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

print("=" * 60)
print("Starting Quantization Experiments")
print("=" * 60)

for width in bit_widths:
    print(f"\n{'='*60}")
    print(f"Testing {width}-bit quantization")
    print(f"{'='*60}")

    # Check if this model already exists
    model_name = f"qat_{width}bit"
    model_path = models_dir / model_name
    cache_key = f"{width}bit"

    if cache_key in cached_results:
        # Load cached results
        print(f"  ✓ Found existing {width}-bit model, loading cached results...")
        ptq_acc = cached_results[cache_key]['ptq_accuracy']
        qat_acc = cached_results[cache_key]['qat_accuracy']
        ptq_accuracies.append(ptq_acc)
        qat_accuracies.append(qat_acc)
        print(f"  ✓ PTQ Accuracy: {ptq_acc:.4f} (cached)")
        print(f"  ✓ QAT Accuracy: {qat_acc:.4f} (cached)")
        print(f"  → Improvement: {qat_acc - ptq_acc:+.4f} ({(qat_acc - ptq_acc)*100:+.2f}%)")
        models_from_cache += 1
        continue

    # Model doesn't exist, train it
    print(f"  No cached model found, training from scratch...")

    # 1. Load fresh model for each experiment
    # Load from Tutorial 2 checkpoint (LoRA fine-tuned model)
    print(f"  Loading model from checkpoint...")
    mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

    # 2. Create quantization config for this bit width
    # Using half bits for fractional part
    frac_width = width // 2

    quantization_config = {
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
                "data_in_width": width,
                "data_in_frac_width": frac_width,
                # weight
                "weight_width": width,
                "weight_frac_width": frac_width,
                # bias
                "bias_width": width,
                "bias_frac_width": frac_width,
            }
        },
    }

    # 3. Apply quantization pass
    print(f"  Applying quantization transform...")
    mg, _ = passes.quantize_transform_pass(
        mg,
        pass_args=quantization_config,
    )

    # 4. Evaluate PTQ (no training)
    print(f"  Evaluating PTQ...")
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # For QAT later
    )

    ptq_results = trainer.evaluate()
    ptq_acc = ptq_results['eval_accuracy']
    ptq_accuracies.append(ptq_acc)
    print(f"  ✓ PTQ Accuracy: {ptq_acc:.4f}")

    # 5. Run QAT (train then evaluate)
    print(f"  Training QAT (1 epoch)...")
    trainer.train()

    qat_results = trainer.evaluate()
    qat_acc = qat_results['eval_accuracy']
    qat_accuracies.append(qat_acc)
    print(f"  ✓ QAT Accuracy: {qat_acc:.4f}")
    print(f"  → Improvement: {qat_acc - ptq_acc:+.4f} ({(qat_acc - ptq_acc)*100:+.2f}%)")

    # 6. Save this model and results
    print(f"  Saving {width}-bit model...")
    mg.export(str(model_path))

    # Update cache
    cached_results[cache_key] = {
        'bit_width': width,
        'ptq_accuracy': float(ptq_acc),
        'qat_accuracy': float(qat_acc),
    }

    # Save cache file after each model
    with open(results_cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)

    print(f"  ✓ Model saved to {model_path}")
    print(f"  ✓ Results cached")
    models_newly_trained += 1

print("\n" + "=" * 60)
print("Experiments Complete!")
print("=" * 60)
print(f"Models loaded from cache: {models_from_cache}")
print(f"Models newly trained: {models_newly_trained}")
print(f"Total models: {len(bit_widths)}")
print("=" * 60)

# 6. Create visualization
print("\nGenerating plot...")
plt.figure(figsize=(10, 6))

plt.plot(bit_widths, ptq_accuracies,
         marker='o', linewidth=2, markersize=8,
         label='PTQ (Post-Training Quantization)',
         color='#e74c3c', linestyle='--')

plt.plot(bit_widths, qat_accuracies,
         marker='s', linewidth=2, markersize=8,
         label='QAT (Quantization-Aware Training)',
         color='#2ecc71', linestyle='-')

# Add horizontal line for FP32 baseline (from Tutorial 2)
fp32_baseline = 0.81396  # From tutorial_2_lora
plt.axhline(y=fp32_baseline, color='gray', linestyle=':',
            linewidth=1.5, label='FP32 Baseline (LoRA)')

# Formatting
plt.xlabel('Fixed-Point Bit Width', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy on IMDb Dataset', fontsize=12, fontweight='bold')
plt.title('Effect of Quantization Precision on BERT Accuracy\nPTQ vs QAT',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(bit_widths)
plt.ylim([0.5, 0.9])  # Adjust based on your results

# Add annotations for improvement at each point
for i, width in enumerate(bit_widths):
    improvement = qat_accuracies[i] - ptq_accuracies[i]
    if improvement > 0.02:  # Only annotate significant improvements
        mid_y = (ptq_accuracies[i] + qat_accuracies[i]) / 2
        plt.annotate(f'+{improvement:.2%}',
                    xy=(width, mid_y),
                    fontsize=8, ha='left',
                    color='darkgreen', fontweight='bold')

plt.tight_layout()
output_dir = Path(__file__).parent / 'lab_1_task_1_results'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'quantization_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved as '{output_file}'")
plt.show()

# 7. Print summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Bit Width':<12} {'PTQ Acc':<12} {'QAT Acc':<12} {'Improvement':<12}")
print("-" * 60)
for i, width in enumerate(bit_widths):
    improvement = qat_accuracies[i] - ptq_accuracies[i]
    print(f"{width:<12} {ptq_accuracies[i]:<12.4f} {qat_accuracies[i]:<12.4f} {improvement:+.4f}")
print("=" * 60)

# 8. Save results for later analysis
results = {
    'bit_widths': bit_widths,
    'ptq_accuracies': ptq_accuracies,
    'qat_accuracies': qat_accuracies,
    'fp32_baseline': fp32_baseline,
}

results_file = output_dir / 'quantization_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to '{results_file}'")

# 9. Key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"Best PTQ accuracy: {max(ptq_accuracies):.4f} at {bit_widths[ptq_accuracies.index(max(ptq_accuracies))]} bits")
print(f"Best QAT accuracy: {max(qat_accuracies):.4f} at {bit_widths[qat_accuracies.index(max(qat_accuracies))]} bits")
print(f"Largest QAT improvement: {max(np.array(qat_accuracies) - np.array(ptq_accuracies)):.4f} at {bit_widths[np.argmax(np.array(qat_accuracies) - np.array(ptq_accuracies))]} bits")
print("=" * 60)

# ============================================================================
# PHASE 2: IMBALANCED PRECISION SWEEP
# ============================================================================
# Vary integer and fractional widths independently to find optimal split.
# Format: Q{integer}.{fractional} where total_width = integer + fractional

print("\n" + "=" * 60)
print("Phase 2: Imbalanced Precision Sweep")
print("=" * 60)

# Define imbalanced configs: (total_width, frac_width)
# integer_bits = total_width - frac_width
imbalanced_configs = [
    # 8-bit total: vary frac from 2 to 6
    (8, 2),   # Q6.2
    (8, 4),   # Q4.4 (balanced)
    (8, 6),   # Q2.6
    # 12-bit total: vary frac from 2 to 10
    (12, 2),  # Q10.2
    (12, 4),  # Q8.4
    (12, 6),  # Q6.6 (balanced)
    (12, 8),  # Q4.8
    (12, 10), # Q2.10
    # 16-bit total: vary frac from 2 to 14
    (16, 2),  # Q14.2
    (16, 4),  # Q12.4
    (16, 8),  # Q8.8 (balanced)
    (16, 12), # Q4.12
    (16, 14), # Q2.14
]

imbalanced_results = {}
imbalanced_cache_file = models_dir / "imbalanced_cache.json"

if imbalanced_cache_file.exists():
    with open(imbalanced_cache_file, 'r') as f:
        imbalanced_cache = json.load(f)
    print(f"Loaded {len(imbalanced_cache)} cached imbalanced results")
else:
    imbalanced_cache = {}

for total_w, frac_w in imbalanced_configs:
    int_w = total_w - frac_w
    label = f"Q{int_w}.{frac_w}"
    cache_key = f"{total_w}_{frac_w}"

    print(f"\n{'='*60}")
    print(f"Testing {label} (total={total_w}, int={int_w}, frac={frac_w})")
    print(f"{'='*60}")

    if cache_key in imbalanced_cache:
        print(f"  Using cached results...")
        imbalanced_results[cache_key] = imbalanced_cache[cache_key]
        print(f"  PTQ: {imbalanced_cache[cache_key]['ptq_accuracy']:.4f}")
        print(f"  QAT: {imbalanced_cache[cache_key]['qat_accuracy']:.4f}")
        continue

    # Load fresh model
    print(f"  Loading model...")
    mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

    # Quantization config with imbalanced widths
    quantization_config = {
        "by": "type",
        "default": {
            "config": {"name": None}
        },
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": total_w,
                "data_in_frac_width": frac_w,
                "weight_width": total_w,
                "weight_frac_width": frac_w,
                "bias_width": total_w,
                "bias_frac_width": frac_w,
            }
        },
    }

    # Apply quantization
    print(f"  Applying quantization...")
    mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)

    # PTQ evaluation
    print(f"  Evaluating PTQ...")
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    ptq_results = trainer.evaluate()
    ptq_acc = ptq_results['eval_accuracy']

    # QAT
    print(f"  Training QAT (1 epoch)...")
    trainer.train()
    qat_results = trainer.evaluate()
    qat_acc = qat_results['eval_accuracy']

    print(f"  PTQ: {ptq_acc:.4f}, QAT: {qat_acc:.4f}")

    result = {
        'total_width': total_w,
        'frac_width': frac_w,
        'int_width': int_w,
        'label': label,
        'ptq_accuracy': float(ptq_acc),
        'qat_accuracy': float(qat_acc),
    }
    imbalanced_results[cache_key] = result
    imbalanced_cache[cache_key] = result

    with open(imbalanced_cache_file, 'w') as f:
        json.dump(imbalanced_cache, f, indent=2)

# Generate heatmap for QAT results
print("\nGenerating imbalanced precision heatmap...")

total_widths_set = sorted(set(tw for tw, _ in imbalanced_configs))
frac_widths_set = sorted(set(fw for _, fw in imbalanced_configs))

# Build heatmap matrices for PTQ and QAT
ptq_matrix = np.full((len(total_widths_set), len(frac_widths_set)), np.nan)
qat_matrix = np.full((len(total_widths_set), len(frac_widths_set)), np.nan)

for cache_key, res in imbalanced_results.items():
    tw, fw = res['total_width'], res['frac_width']
    row = total_widths_set.index(tw)
    col = frac_widths_set.index(fw)
    ptq_matrix[row, col] = res['ptq_accuracy']
    qat_matrix[row, col] = res['qat_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, matrix, title in [
    (axes[0], ptq_matrix, 'PTQ Accuracy'),
    (axes[1], qat_matrix, 'QAT Accuracy'),
]:
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.85)
    ax.set_xticks(range(len(frac_widths_set)))
    ax.set_xticklabels(frac_widths_set)
    ax.set_yticks(range(len(total_widths_set)))
    ax.set_yticklabels(total_widths_set)
    ax.set_xlabel('Fractional Width (bits)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Width (bits)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(len(total_widths_set)):
        for j in range(len(frac_widths_set)):
            val = matrix[i, j]
            if not np.isnan(val):
                int_w = total_widths_set[i] - frac_widths_set[j]
                if int_w > 0:
                    color = 'white' if val < 0.65 else 'black'
                    ax.text(j, i, f'{val:.3f}\nQ{int_w}.{frac_widths_set[j]}',
                            ha='center', va='center', fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Effect of Integer vs Fractional Bit Allocation on Accuracy',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

imbalanced_plot = output_dir / 'imbalanced_precision_heatmap.png'
plt.savefig(imbalanced_plot, dpi=300, bbox_inches='tight')
print(f"Heatmap saved as '{imbalanced_plot}'")
plt.show()

# Print imbalanced summary
print("\n" + "=" * 60)
print("IMBALANCED PRECISION SUMMARY")
print("=" * 60)
print(f"{'Config':<10} {'Total':<8} {'Int':<6} {'Frac':<6} {'PTQ':<10} {'QAT':<10}")
print("-" * 60)
for cache_key, res in sorted(imbalanced_results.items()):
    print(f"{res['label']:<10} {res['total_width']:<8} {res['int_width']:<6} "
          f"{res['frac_width']:<6} {res['ptq_accuracy']:<10.4f} {res['qat_accuracy']:<10.4f}")
print("=" * 60)

# Find optimal split for each total width
print("\nOptimal frac_width per total width (QAT):")
for tw in total_widths_set:
    best_acc = 0
    best_fw = 0
    for ck, res in imbalanced_results.items():
        if res['total_width'] == tw and res['qat_accuracy'] > best_acc:
            best_acc = res['qat_accuracy']
            best_fw = res['frac_width']
    print(f"  {tw}-bit: best frac_width={best_fw} (Q{tw-best_fw}.{best_fw}), QAT={best_acc:.4f}")

# Save imbalanced results
imbalanced_save = {
    'configs': [
        imbalanced_results[f"{tw}_{fw}"]
        for tw, fw in imbalanced_configs
        if f"{tw}_{fw}" in imbalanced_results
    ],
}
imbalanced_file = output_dir / 'imbalanced_precision_results.json'
with open(imbalanced_file, 'w') as f:
    json.dump(imbalanced_save, f, indent=2)
print(f"\nImbalanced results saved to '{imbalanced_file}'")

# Update overall results with imbalanced data
results['imbalanced'] = imbalanced_save['configs']
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# 10. Save the best QAT model for Task 2
print("\n" + "=" * 60)
print("Saving Best Model for Task 2")
print("=" * 60)

# Find best QAT model
best_qat_idx = qat_accuracies.index(max(qat_accuracies))
best_bit_width = bit_widths[best_qat_idx]
best_qat_acc = qat_accuracies[best_qat_idx]

print(f"Best model: {best_bit_width}-bit QAT with accuracy {best_qat_acc:.4f}")

# Copy the best model from task_1_models directory
import shutil
source_model_name = f"qat_{best_bit_width}bit"
source_path = models_dir / source_model_name
dest_path = Path.home() / "task_1_best_model"

# Copy both .pt and .mz files
for ext in ['.pt', '.mz']:
    source_file = Path(str(source_path) + ext)
    dest_file = Path(str(dest_path) + ext)

    if source_file.exists():
        shutil.copy2(source_file, dest_file)
        print(f"  ✓ Copied {source_file.name}")
    else:
        print(f"  ⚠ Warning: {source_file.name} not found")

print(f"✓ Best model saved to {dest_path}")
print(f"  This model will be used as the base for Task 2 (pruning)")
print("=" * 60)
