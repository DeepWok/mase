"""
Lab 1 Task 1: Quantization Precision Analysis

Two experiments:
1. Balanced sweep: Total bit widths from 4 to 32 (frac = width//2), PTQ vs QAT.
2. Imbalanced sweep: For total widths {8, 12, 16}, vary frac_width independently
   (e.g. Q4.8 = 4 integer bits, 8 fractional bits). Generates a heatmap.

Features:
- Caches results to ~/task_1_models/ for resume capability
- Saves the best QAT model to ~/task_1_best_model for use in Task 2
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import shutil
from pathlib import Path
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

bit_widths = [4, 6, 8, 12, 16, 24, 32]

ptq_accuracies = []
qat_accuracies = []

models_dir = Path.home() / "task_1_models"
models_dir.mkdir(exist_ok=True)
results_cache_file = models_dir / "results_cache.json"

if results_cache_file.exists():
    with open(results_cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for {len(cached_results)} models")
else:
    cached_results = {}

models_from_cache = 0
models_newly_trained = 0

print("Loading dataset...")
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

for width in bit_widths:
    model_name = f"qat_{width}bit"
    model_path = models_dir / model_name
    cache_key = f"{width}bit"

    if cache_key in cached_results:
        ptq_acc = cached_results[cache_key]['ptq_accuracy']
        qat_acc = cached_results[cache_key]['qat_accuracy']
        ptq_accuracies.append(ptq_acc)
        qat_accuracies.append(qat_acc)
        print(f"  {width}-bit: PTQ={ptq_acc:.4f}, QAT={qat_acc:.4f} (cached)")
        models_from_cache += 1
        continue

    print(f"  Training {width}-bit model...")
    mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

    frac_width = width // 2
    quantization_config = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": width,
                "data_in_frac_width": frac_width,
                "weight_width": width,
                "weight_frac_width": frac_width,
                "bias_width": width,
                "bias_frac_width": frac_width,
            }
        },
    }

    mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)

    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    ptq_results = trainer.evaluate()
    ptq_acc = ptq_results['eval_accuracy']
    ptq_accuracies.append(ptq_acc)

    trainer.train()
    qat_results = trainer.evaluate()
    qat_acc = qat_results['eval_accuracy']
    qat_accuracies.append(qat_acc)

    print(f"  {width}-bit: PTQ={ptq_acc:.4f}, QAT={qat_acc:.4f}, gap={qat_acc - ptq_acc:+.4f}")

    mg.export(str(model_path))
    cached_results[cache_key] = {
        'bit_width': width,
        'ptq_accuracy': float(ptq_acc),
        'qat_accuracy': float(qat_acc),
    }
    with open(results_cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)
    models_newly_trained += 1

print(f"\nBalanced sweep done: {models_from_cache} cached, {models_newly_trained} trained")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bit_widths, ptq_accuracies,
         marker='o', linewidth=2, markersize=8,
         label='PTQ (Post-Training Quantization)',
         color='#e74c3c', linestyle='--')
plt.plot(bit_widths, qat_accuracies,
         marker='s', linewidth=2, markersize=8,
         label='QAT (Quantization-Aware Training)',
         color='#2ecc71', linestyle='-')

fp32_baseline = 0.81396
plt.axhline(y=fp32_baseline, color='gray', linestyle=':',
            linewidth=1.5, label='FP32 Baseline (LoRA)')

plt.xlabel('Fixed-Point Bit Width', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy on IMDb Dataset', fontsize=12, fontweight='bold')
plt.title('Effect of Quantization Precision on BERT Accuracy\nPTQ vs QAT',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(bit_widths)
plt.ylim([0.5, 0.9])

for i, width in enumerate(bit_widths):
    improvement = qat_accuracies[i] - ptq_accuracies[i]
    if improvement > 0.02:
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
print(f"Plot saved as '{output_file}'")
plt.show()

# Summary
print(f"\n{'Bit Width':<12} {'PTQ Acc':<12} {'QAT Acc':<12} {'Improvement':<12}")
print("-" * 48)
for i, width in enumerate(bit_widths):
    improvement = qat_accuracies[i] - ptq_accuracies[i]
    print(f"{width:<12} {ptq_accuracies[i]:<12.4f} {qat_accuracies[i]:<12.4f} {improvement:+.4f}")

# Save results
results = {
    'bit_widths': bit_widths,
    'ptq_accuracies': ptq_accuracies,
    'qat_accuracies': qat_accuracies,
    'fp32_baseline': fp32_baseline,
}
results_file = output_dir / 'quantization_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to '{results_file}'")

print(f"\nBest PTQ: {max(ptq_accuracies):.4f} at {bit_widths[ptq_accuracies.index(max(ptq_accuracies))]} bits")
print(f"Best QAT: {max(qat_accuracies):.4f} at {bit_widths[qat_accuracies.index(max(qat_accuracies))]} bits")

# Phase 2: Imbalanced Precision Sweep
print("\nStarting imbalanced precision sweep...")

imbalanced_configs = [
    (8, 2), (8, 4), (8, 6),
    (12, 2), (12, 4), (12, 6), (12, 8), (12, 10),
    (16, 2), (16, 4), (16, 8), (16, 12), (16, 14),
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

    if cache_key in imbalanced_cache:
        imbalanced_results[cache_key] = imbalanced_cache[cache_key]
        print(f"  {label}: PTQ={imbalanced_cache[cache_key]['ptq_accuracy']:.4f}, QAT={imbalanced_cache[cache_key]['qat_accuracy']:.4f} (cached)")
        continue

    print(f"  Training {label} (total={total_w}, int={int_w}, frac={frac_w})...")
    mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

    quantization_config = {
        "by": "type",
        "default": {"config": {"name": None}},
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

    mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)

    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    ptq_results = trainer.evaluate()
    ptq_acc = ptq_results['eval_accuracy']

    trainer.train()
    qat_results = trainer.evaluate()
    qat_acc = qat_results['eval_accuracy']

    print(f"  {label}: PTQ={ptq_acc:.4f}, QAT={qat_acc:.4f}")

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

# Generate heatmap
print("\nGenerating imbalanced precision heatmap...")
total_widths_set = sorted(set(tw for tw, _ in imbalanced_configs))
frac_widths_set = sorted(set(fw for _, fw in imbalanced_configs))

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
    flipped = matrix[::-1]
    im = ax.imshow(flipped, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.85)
    ax.set_xticks(range(len(frac_widths_set)))
    ax.set_xticklabels(frac_widths_set)
    ax.set_yticks(range(len(total_widths_set)))
    ax.set_yticklabels(total_widths_set[::-1])
    ax.set_xlabel('Fractional Width (bits)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Width (bits)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')

    for i in range(len(total_widths_set)):
        for j in range(len(frac_widths_set)):
            val = flipped[i, j]
            if not np.isnan(val):
                orig_row = len(total_widths_set) - 1 - i
                int_w = total_widths_set[orig_row] - frac_widths_set[j]
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

# Imbalanced summary
print(f"\n{'Config':<10} {'Total':<8} {'Int':<6} {'Frac':<6} {'PTQ':<10} {'QAT':<10}")
print("-" * 50)
for cache_key, res in sorted(imbalanced_results.items()):
    print(f"{res['label']:<10} {res['total_width']:<8} {res['int_width']:<6} "
          f"{res['frac_width']:<6} {res['ptq_accuracy']:<10.4f} {res['qat_accuracy']:<10.4f}")

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
print(f"Imbalanced results saved to '{imbalanced_file}'")

results['imbalanced'] = imbalanced_save['configs']
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save the best QAT model for Task 2
best_qat_idx = qat_accuracies.index(max(qat_accuracies))
best_bit_width = bit_widths[best_qat_idx]
best_qat_acc = qat_accuracies[best_qat_idx]
print(f"\nBest model: {best_bit_width}-bit QAT with accuracy {best_qat_acc:.4f}")

source_model_name = f"qat_{best_bit_width}bit"
source_path = models_dir / source_model_name
dest_path = Path.home() / "task_1_best_model"

for ext in ['.pt', '.mz']:
    source_file = Path(str(source_path) + ext)
    dest_file = Path(str(dest_path) + ext)
    if source_file.exists():
        shutil.copy2(source_file, dest_file)
        print(f"  Copied {source_file.name}")

print(f"Best model saved to {dest_path} for Task 2 (pruning)")
