"""
Lab 2 Task 2: Compression-Aware NAS

This script compares three experimental conditions:
1. Baseline: NAS without compression
2. Compress-Only: NAS with compression (no post-compression training)
3. Compress+Finetune: NAS with compression and 1 epoch post-compression training

The script uses TPESampler for all conditions to ensure fair comparison.

Fix notes:
- Uses module-level quantization (deepsetattr + LinearInteger) instead of
  graph-level quantize_transform_pass, avoiding MaseGraph metadata init issues.
- No need for KeywordToPositionalWrapper or special evaluate_compressed_model.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import optuna
from pathlib import Path
from optuna.samplers import TPESampler
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.nn.quantized.modules.linear import LinearInteger
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and dataset config
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# Search space (same as Task 1)
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
}

# Quantization config for module-level replacement
QUANTIZATION_CONFIG = {
    "data_in_width": 8,
    "data_in_frac_width": 4,
    "weight_width": 8,
    "weight_frac_width": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
}

# Experiment config
N_TRIALS = 10  # Increase to 50-100 for production

# ============================================================================
# CACHING INFRASTRUCTURE
# ============================================================================

results_dir = Path.home() / "lab_2_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_2_cache.json"

# Load existing cache
if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for conditions: {list(cached_results.keys())}")
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
# HELPER FUNCTIONS
# ============================================================================

def construct_model(trial):
    """Construct a BERT model with sampled hyperparameters."""
    config = AutoConfig.from_pretrained(checkpoint)

    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    model = AutoModelForSequenceClassification.from_config(config)
    return model


def apply_quantization(model):
    """Apply integer quantization to all Linear layers using module-level replacement.

    This avoids MaseGraph metadata initialization issues by directly replacing
    nn.Linear layers with LinearInteger layers and copying weights.
    """
    for name, layer in list(model.named_modules()):
        if isinstance(layer, torch.nn.Linear):
            new_layer = LinearInteger(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                config=copy.deepcopy(QUANTIZATION_CONFIG),
            )
            # Copy weights from original layer
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            # Replace layer in model
            deepsetattr(model, name, new_layer)
    return model

# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================

def objective_baseline(trial):
    """Baseline: Train and evaluate without compression."""
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
    accuracy = eval_results["eval_accuracy"]
    print(f"  Trial {trial.number} [Baseline]: accuracy={accuracy:.4f}")
    return accuracy


def objective_compress_no_finetune(trial):
    """Compression without post-compression training.

    1. Construct model with sampled architecture
    2. Train for 1 epoch
    3. Apply INT8 quantization (module-level)
    4. Evaluate (no additional training after compression)
    """
    model = construct_model(trial)

    # Initial training
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()

    # Apply quantization (module-level, no MaseGraph needed)
    model = apply_quantization(model)

    # Evaluate compressed model directly (no wrapper needed)
    eval_trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    eval_results = eval_trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    print(f"  Trial {trial.number} [Compress-Only]: accuracy={accuracy:.4f}")
    return accuracy


def objective_compress_with_finetune(trial):
    """Compression with post-compression training.

    1. Construct model with sampled architecture
    2. Train for 1 epoch
    3. Apply INT8 quantization (module-level)
    4. Fine-tune for 1 additional epoch after compression
    5. Evaluate
    """
    model = construct_model(trial)

    # Initial training
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()

    # Apply quantization (module-level, no MaseGraph needed)
    model = apply_quantization(model)

    # Post-compression fine-tuning (1 additional epoch)
    finetune_trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    finetune_trainer.train()

    # Evaluate
    eval_results = finetune_trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    print(f"  Trial {trial.number} [Compress+Finetune]: accuracy={accuracy:.4f}")
    return accuracy

# ============================================================================
# SEARCH EXECUTION
# ============================================================================

def run_condition(condition_name, objective_fn, n_trials):
    """Run NAS for a specific experimental condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition_name}")
    print(f"{'='*60}")

    # Check cache
    if condition_name in cached_results:
        print(f"  Found cached results, skipping...")
        return cached_results[condition_name]

    # Create study with TPESampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lab2-task2-{condition_name}",
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials, timeout=60*60*24)

    # Calculate best-so-far
    best_so_far = []
    for trial_num in range(len(study.trials)):
        accuracies = [t.value for t in study.trials[:trial_num+1] if t.value is not None]
        best_so_far.append(max(accuracies) if accuracies else 0)

    # Cache results
    result = {
        "trial_accuracies": [t.value for t in study.trials],
        "best_so_far": best_so_far,
        "best_accuracy": study.best_value,
        "best_params": {k: v for k, v in study.best_params.items()},
    }
    cached_results[condition_name] = result
    save_cache()

    print(f"  Best accuracy: {study.best_value:.4f}")
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define conditions
    conditions = {
        "Baseline": objective_baseline,
        "Compress-Only": objective_compress_no_finetune,
        "Compress+Finetune": objective_compress_with_finetune,
    }

    # Run experiments
    results = {}
    for name, objective_fn in conditions.items():
        results[name] = run_condition(name, objective_fn, N_TRIALS)

    print("\n" + "="*60)
    print("All conditions complete!")
    print("="*60)

    # ====================================================================
    # PLOT GENERATION
    # ====================================================================

    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))

    colors = {
        'Baseline': '#2ecc71',
        'Compress-Only': '#e74c3c',
        'Compress+Finetune': '#3498db',
    }
    markers = {
        'Baseline': 'o',
        'Compress-Only': 's',
        'Compress+Finetune': '^',
    }
    linestyles = {
        'Baseline': '-',
        'Compress-Only': '--',
        'Compress+Finetune': '-',
    }

    for name, result in results.items():
        trials = list(range(1, len(result["best_so_far"]) + 1))
        plt.plot(
            trials,
            result["best_so_far"],
            marker=markers[name],
            linewidth=2,
            markersize=6,
            label=f'{name} (best: {result["best_accuracy"]:.4f})',
            color=colors[name],
            linestyle=linestyles[name],
        )

    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
    plt.title(
        'Compression-Aware NAS: Effect of Compression on Search Quality',
        fontsize=14, fontweight='bold', pad=20,
    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent / 'lab_2_task_2_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'compression_nas_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

    # ====================================================================
    # RESULTS ANALYSIS
    # ====================================================================

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Condition':<25} {'Best Accuracy':<15} {'vs Baseline':<15}")
    print("-"*60)

    baseline_acc = results["Baseline"]["best_accuracy"]
    for name, result in results.items():
        diff = result["best_accuracy"] - baseline_acc
        diff_str = f"{diff:+.4f}" if name != "Baseline" else "-"
        print(f"{name:<25} {result['best_accuracy']:<15.4f} {diff_str:<15}")
    print("="*60)

    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    compress_only_acc = results["Compress-Only"]["best_accuracy"]
    compress_finetune_acc = results["Compress+Finetune"]["best_accuracy"]

    print(f"Compression degradation (no finetune): {compress_only_acc - baseline_acc:+.4f}")
    print(f"Compression degradation (with finetune): {compress_finetune_acc - baseline_acc:+.4f}")
    print(f"Finetune recovery: {compress_finetune_acc - compress_only_acc:+.4f}")

    if compress_finetune_acc >= baseline_acc * 0.95:
        print("\nConclusion: Post-compression training successfully recovers accuracy!")
    else:
        print("\nConclusion: Significant accuracy loss from compression.")
    print("="*60)

    # Save final results
    final_results = {
        "n_trials": N_TRIALS,
        "compression_config": {
            "quantization": QUANTIZATION_CONFIG,
            "method": "module-level LinearInteger replacement",
            "note": "Applied INT8 quantization to all Linear layers",
        },
        "conditions": results,
    }
    results_file = output_dir / 'compression_nas_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to '{results_file}'")