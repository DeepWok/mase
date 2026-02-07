"""
Lab 2 Task 2: Compression-Aware NAS

Compares three experimental conditions:
1. Baseline: NAS without compression
2. Compress-Only: NAS with compression (no post-compression training)
3. Compress+Finetune: NAS with compression and 1 epoch post-compression training

Uses TPESampler for all conditions. Module-level quantization via
deepsetattr + LinearInteger to avoid MaseGraph metadata init issues.
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

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
}

QUANTIZATION_CONFIG = {
    "data_in_width": 8,
    "data_in_frac_width": 4,
    "weight_width": 8,
    "weight_frac_width": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
}

N_TRIALS = 10

results_dir = Path.home() / "lab_2_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_2_cache.json"

if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for conditions: {list(cached_results.keys())}")
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


def apply_quantization(model):
    """Apply integer quantization to all Linear layers using module-level replacement."""
    for name, layer in list(model.named_modules()):
        if isinstance(layer, torch.nn.Linear):
            new_layer = LinearInteger(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                config=copy.deepcopy(QUANTIZATION_CONFIG),
            )
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            deepsetattr(model, name, new_layer)
    return model


def objective_baseline(trial):
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
    model = construct_model(trial)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()
    model = apply_quantization(model)
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
    model = construct_model(trial)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()
    model = apply_quantization(model)
    finetune_trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    finetune_trainer.train()
    eval_results = finetune_trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    print(f"  Trial {trial.number} [Compress+Finetune]: accuracy={accuracy:.4f}")
    return accuracy


def run_condition(condition_name, objective_fn, n_trials):
    if condition_name in cached_results:
        print(f"  {condition_name}: using cached results")
        return cached_results[condition_name]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lab2-task2-{condition_name}",
        sampler=sampler,
    )
    study.optimize(objective_fn, n_trials=n_trials, timeout=60*60*24)

    best_so_far = []
    for trial_num in range(len(study.trials)):
        accuracies = [t.value for t in study.trials[:trial_num+1] if t.value is not None]
        best_so_far.append(max(accuracies) if accuracies else 0)

    result = {
        "trial_accuracies": [t.value for t in study.trials],
        "best_so_far": best_so_far,
        "best_accuracy": study.best_value,
        "best_params": {k: v for k, v in study.best_params.items()},
    }
    cached_results[condition_name] = result
    save_cache()

    print(f"  {condition_name} best accuracy: {study.best_value:.4f}")
    return result


if __name__ == "__main__":
    conditions = {
        "Baseline": objective_baseline,
        "Compress-Only": objective_compress_no_finetune,
        "Compress+Finetune": objective_compress_with_finetune,
    }

    results = {}
    for name, objective_fn in conditions.items():
        results[name] = run_condition(name, objective_fn, N_TRIALS)

    # Plot
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

    output_dir = Path(__file__).parent / 'lab_2_task_2_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'compression_nas_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

    # Summary
    print(f"\n{'Condition':<25} {'Best Accuracy':<15} {'vs Baseline':<15}")
    print("-" * 55)
    baseline_acc = results["Baseline"]["best_accuracy"]
    for name, result in results.items():
        diff = result["best_accuracy"] - baseline_acc
        diff_str = f"{diff:+.4f}" if name != "Baseline" else "-"
        print(f"{name:<25} {result['best_accuracy']:<15.4f} {diff_str:<15}")

    compress_only_acc = results["Compress-Only"]["best_accuracy"]
    compress_finetune_acc = results["Compress+Finetune"]["best_accuracy"]
    print(f"\nCompression degradation (no finetune): {compress_only_acc - baseline_acc:+.4f}")
    print(f"Compression degradation (with finetune): {compress_finetune_acc - baseline_acc:+.4f}")
    print(f"Finetune recovery: {compress_finetune_acc - compress_only_acc:+.4f}")

    if compress_finetune_acc >= baseline_acc * 0.95:
        print("Conclusion: Post-compression training successfully recovers accuracy!")
    else:
        print("Conclusion: Significant accuracy loss from compression.")

    # Save results
    final_results = {
        "n_trials": N_TRIALS,
        "compression_config": {
            "quantization": QUANTIZATION_CONFIG,
            "method": "module-level LinearInteger replacement",
        },
        "conditions": results,
    }
    results_file = output_dir / 'compression_nas_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to '{results_file}'")
