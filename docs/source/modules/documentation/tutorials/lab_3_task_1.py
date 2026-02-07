"""
Lab 3 Task 1: Per-Layer Mixed Precision Search

Implements per-layer quantization type search using Optuna TPESampler. Each Linear layer
independently samples from 4 quantization types (FP32, LinearInteger, LinearMinifloatIEEE,
LinearLog) with type-specific hyperparameters.

Features:
- Caches results to ~/lab_3_results/task_1_cache.json for resume capability
- Tests 20 trials with TPESampler
- Generates plot: trials vs best accuracy
- Saves detailed results to mixed_precision_results.json
"""

import json
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import TPESampler

from transformers import AutoModelForSequenceClassification

import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatIEEE,
    LinearLog,
)
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

QUANTIZATION_TYPES = [
    torch.nn.Linear,
    LinearInteger,
    LinearMinifloatIEEE,
    LinearLog,
]

WIDTH_CHOICES = [8, 16, 32]
FRAC_WIDTH_CHOICES = [2, 4, 8]
EXPONENT_WIDTH_CHOICES = [3, 4, 5]
EXPONENT_BIAS_CHOICES = [7, 15]

N_TRIALS = 20

results_dir = Path.home() / "lab_3_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_1_cache.json"

if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Loaded {len(cached_results.get('trials', []))} cached trials")
else:
    cached_results = {"trials": [], "best_so_far": [], "best_config": None}

def save_cache():
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)


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
        "data_in_exponent_bias": 7,
        "weight_width": weight_width,
        "weight_exponent_width": weight_exp_width,
        "weight_exponent_bias": 7,
        "bias_width": 16,
        "bias_exponent_width": 5,
        "bias_exponent_bias": 15,
    }

def get_log_config(trial, layer_name):
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
    if layer_cls == torch.nn.Linear:
        return None
    elif layer_cls == LinearInteger:
        return get_integer_config(trial, layer_name)
    elif layer_cls == LinearMinifloatIEEE:
        return get_minifloat_ieee_config(trial, layer_name)
    elif layer_cls == LinearLog:
        return get_log_config(trial, layer_name)
    else:
        raise ValueError(f"Unknown layer type: {layer_cls}")


def construct_model(trial, base_model):
    trial_model = deepcopy(base_model)
    layer_configs = {}

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            layer_type_idx = trial.suggest_int(
                f"{name}_type_idx", 0, len(QUANTIZATION_TYPES) - 1
            )
            new_layer_cls = QUANTIZATION_TYPES[layer_type_idx]

            layer_configs[name] = {
                "type": new_layer_cls.__name__,
                "type_idx": layer_type_idx,
            }

            if new_layer_cls == torch.nn.Linear:
                continue

            config = get_config_for_type(new_layer_cls, trial, name)
            layer_configs[name]["config"] = config

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
                "config": config,
            }
            new_layer = new_layer_cls(**kwargs)

            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()

            deepsetattr(trial_model, name, new_layer)

    trial.set_user_attr("layer_configs", layer_configs)
    return trial_model


def objective(trial):
    try:
        model = construct_model(trial, base_model)
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
        print(f"Trial {trial.number}: accuracy={accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_search(n_trials):
    if cached_results["trials"] and len(cached_results["trials"]) >= n_trials:
        print("Using cached results...")
        return cached_results

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name="lab3-mixed-precision",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, timeout=60*60*24)

    best_so_far = []
    for trial_num in range(len(study.trials)):
        accuracies = [t.value for t in study.trials[:trial_num+1] if t.value is not None]
        best_so_far.append(max(accuracies) if accuracies else 0)

    results = {
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": {k: str(v) for k, v in t.params.items()},
                "layer_configs": t.user_attrs.get("layer_configs", {}),
            }
            for t in study.trials
        ],
        "best_so_far": best_so_far,
        "best_accuracy": study.best_value,
        "best_params": {k: str(v) for k, v in study.best_params.items()},
        "best_trial_number": study.best_trial.number,
    }

    cached_results.update(results)
    save_cache()

    print(f"Best accuracy: {study.best_value:.4f} (trial {study.best_trial.number})")
    return results


def generate_plot(results):
    plt.figure(figsize=(10, 6))
    trials = list(range(1, len(results["best_so_far"]) + 1))
    plt.plot(
        trials,
        results["best_so_far"],
        marker='o',
        linewidth=2,
        markersize=6,
        color='#2ecc71',
        label=f'TPESampler (best: {results["best_accuracy"]:.4f})'
    )
    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
    plt.title('Lab 3 Task 1: Per-Layer Mixed Precision Search\n(TPESampler)',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_dir = Path(__file__).parent / 'lab_3_task_1_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'mixed_precision_search.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()


def print_summary(results):
    print(f"\nTotal trials: {len(results['trials'])}")
    print(f"Best accuracy: {results['best_accuracy']:.4f} (trial {results['best_trial_number']})")
    print("\nBest configuration (layer types):")
    best_trial = results['trials'][results['best_trial_number']]
    for layer_name, config in best_trial.get('layer_configs', {}).items():
        print(f"  {layer_name}: {config['type']}")


if __name__ == "__main__":
    print("Loading dataset...")
    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )

    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    )

    results = run_search(N_TRIALS)
    generate_plot(results)
    print_summary(results)

    output_dir = Path(__file__).parent / 'lab_3_task_1_results'
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / 'mixed_precision_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{results_file}'")
