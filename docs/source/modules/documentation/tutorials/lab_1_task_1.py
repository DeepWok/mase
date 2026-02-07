"""
Lab 1 Task 1: Quantization Precision Analysis

Explore a range of fixed point widths from 4 to 32 bits and compare
PTQ (Post-Training Quantization) vs QAT (Quantization-Aware Training)
accuracy on the IMDb sentiment analysis dataset.

Features:
- Automatically saves each trained model to ~/task_1_models/qat_{width}bit
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
