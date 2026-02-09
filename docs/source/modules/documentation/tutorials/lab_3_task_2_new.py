"""
Lab 3 Task 2 (Extended): Sampler Comparison for Mixed Precision Search

Compare RandomSampler vs TPESampler for per-layer mixed precision type search.
Tests 8 quantization types: FP32, LinearInteger, LinearMinifloatIEEE, LinearLog,
LinearMinifloatDenorm, LinearBinary, LinearBlockFP, LinearBlockLog.

Note: LinearTernary is excluded because the ternary quantizer in Mase assumes 4D
tensors (Conv2d), making it incompatible with 2D Linear weights.
LinearBlockMinifloat is excluded due to a bug in BlockMinifloatQuantize.backward()
which has extra positional arguments not provided by PyTorch autograd.

Features:
- Caches results to ~/lab_3_results/task_2_new_cache.json for resume capability
- Tests 100 trials per sampler
- Generates comparison plot: trials vs best accuracy for both samplers
- Saves detailed results to sampler_comparison_results.json
"""

import json
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import RandomSampler, TPESampler

from transformers import AutoModelForSequenceClassification

import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatIEEE,
    LinearLog,
    LinearMinifloatDenorm,
    LinearBinary,
    LinearBlockFP,
    LinearBlockLog,
)
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

QUANTIZATION_TYPES = [
    torch.nn.Linear,        # 0: FP32
    LinearInteger,           # 1
    LinearMinifloatIEEE,     # 2
    LinearLog,               # 3
    LinearMinifloatDenorm,   # 4
    LinearBinary,            # 5
    LinearBlockFP,           # 6
    LinearBlockLog,          # 7
]

WIDTH_CHOICES = [8, 16, 32]
FRAC_WIDTH_CHOICES = [2, 4, 8]
EXPONENT_WIDTH_CHOICES = [3, 4, 5]
EXPONENT_BIAS_CHOICES = [7, 15]
BLOCK_SIZE_CHOICES = [8, 16, 32]
EXPONENT_BIAS_WIDTH_CHOICES = [2, 3, 4]

N_TRIALS_PER_SAMPLER = 100

results_dir = Path.home() / "lab_3_results"
results_dir.mkdir(exist_ok=True)
cache_file = results_dir / "task_2_new_cache.json"

if cache_file.exists():
    with open(cache_file, 'r') as f:
        cached_results = json.load(f)
    print(f"Found cached results for samplers: {list(cached_results.keys())}")
else:
    cached_results = {}

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

def get_minifloat_denorm_config(trial, layer_name):
    data_in_width = trial.suggest_categorical(
        f"{layer_name}_mfd_data_in_width", WIDTH_CHOICES
    )
    data_in_exp_width = trial.suggest_categorical(
        f"{layer_name}_mfd_data_in_exp_width", EXPONENT_WIDTH_CHOICES
    )
    weight_width = trial.suggest_categorical(
        f"{layer_name}_mfd_weight_width", WIDTH_CHOICES
    )
    weight_exp_width = trial.suggest_categorical(
        f"{layer_name}_mfd_weight_exp_width", EXPONENT_WIDTH_CHOICES
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

def get_binary_config(trial, layer_name):
    # bipolar must be True for Linear layers: the non-bipolar path (BinaryZeroScaled)
    # assumes 4D tensors and crashes on 2D Linear weights.
    stochastic = trial.suggest_categorical(f"{layer_name}_binary_stochastic", [True, False])
    return {
        "data_in_width": 1,
        "data_in_stochastic": stochastic,
        "data_in_bipolar": True,
        "weight_width": 1,
        "weight_stochastic": stochastic,
        "weight_bipolar": True,
        "bias_width": 8,
        "bias_stochastic": False,
        "bias_bipolar": True,
    }

def get_block_fp_config(trial, layer_name):
    data_in_width = trial.suggest_categorical(f"{layer_name}_bfp_data_in_width", WIDTH_CHOICES)
    data_in_exp_width = trial.suggest_categorical(f"{layer_name}_bfp_data_in_exp_width", EXPONENT_WIDTH_CHOICES)
    weight_width = trial.suggest_categorical(f"{layer_name}_bfp_weight_width", WIDTH_CHOICES)
    weight_exp_width = trial.suggest_categorical(f"{layer_name}_bfp_weight_exp_width", EXPONENT_WIDTH_CHOICES)
    block_size = trial.suggest_categorical(f"{layer_name}_bfp_block_size", BLOCK_SIZE_CHOICES)
    return {
        "data_in_width": data_in_width,
        "data_in_exponent_width": data_in_exp_width,
        "data_in_exponent_bias": 7,
        "data_in_block_size": [1, block_size],
        "weight_width": weight_width,
        "weight_exponent_width": weight_exp_width,
        "weight_exponent_bias": 7,
        "weight_block_size": [1, block_size],
        "bias_width": 16,
        "bias_exponent_width": 5,
        "bias_exponent_bias": 15,
        "bias_block_size": [block_size],
    }

def get_block_log_config(trial, layer_name):
    data_in_width = trial.suggest_categorical(f"{layer_name}_blog_data_in_width", WIDTH_CHOICES)
    data_in_exp_bias_width = trial.suggest_categorical(f"{layer_name}_blog_data_in_exp_bias_width", EXPONENT_BIAS_WIDTH_CHOICES)
    weight_width = trial.suggest_categorical(f"{layer_name}_blog_weight_width", WIDTH_CHOICES)
    weight_exp_bias_width = trial.suggest_categorical(f"{layer_name}_blog_weight_exp_bias_width", EXPONENT_BIAS_WIDTH_CHOICES)
    block_size = trial.suggest_categorical(f"{layer_name}_blog_block_size", BLOCK_SIZE_CHOICES)
    return {
        "data_in_width": data_in_width,
        "data_in_exponent_bias_width": data_in_exp_bias_width,
        "data_in_block_size": [1, block_size],
        "weight_width": weight_width,
        "weight_exponent_bias_width": weight_exp_bias_width,
        "weight_block_size": [1, block_size],
        "bias_width": 16,
        "bias_exponent_bias_width": 3,
        "bias_block_size": [block_size],
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
    elif layer_cls == LinearMinifloatDenorm:
        return get_minifloat_denorm_config(trial, layer_name)
    elif layer_cls == LinearBinary:
        return get_binary_config(trial, layer_name)
    elif layer_cls == LinearBlockFP:
        return get_block_fp_config(trial, layer_name)
    elif layer_cls == LinearBlockLog:
        return get_block_log_config(trial, layer_name)
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
        study_name=f"lab3-task2-new-{sampler_name}-batch2",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=remaining, timeout=60*60*12)

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
        best_params = {k: str(v) for k, v in study.best_params.items()}

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

    samplers = {
        "RandomSampler": RandomSampler(seed=42),
        "TPESampler": TPESampler(seed=42),
    }

    results = {}
    for name, sampler in samplers.items():
        results[name] = run_search(name, sampler, N_TRIALS_PER_SAMPLER)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = {'RandomSampler': '#3498db', 'TPESampler': '#2ecc71'}
    markers = {'RandomSampler': 's', 'TPESampler': '^'}

    for name, result in results.items():
        trials = list(range(1, len(result["best_so_far"]) + 1))
        plt.plot(
            trials,
            result["best_so_far"],
            marker=markers[name],
            linewidth=2,
            markersize=6,
            color=colors[name],
            label=f'{name} (best: {result["best_accuracy"]:.4f})'
        )

    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy So Far', fontsize=12, fontweight='bold')
    plt.title('Lab 3 Task 2 Extended: Sampler Comparison for 10-Type Mixed Precision Search',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_dir = Path(__file__).parent / 'lab_3_task_2_new_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'sampler_comparison_mixed_precision.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

    # Summary
    print(f"\n{'Sampler':<20} {'Best Accuracy':<15} {'Trials to 95%':<15}")
    print("-" * 50)
    for sampler_name, result in results.items():
        target = result["best_accuracy"] * 0.95
        trials_to_95 = None
        for i, acc in enumerate(result["best_so_far"]):
            if acc >= target:
                trials_to_95 = i + 1
                break
        trials_str = str(trials_to_95) if trials_to_95 else "N/A"
        print(f"{sampler_name:<20} {result['best_accuracy']:<15.4f} {trials_str:<15}")

    best_sampler = max(results.items(), key=lambda x: x[1]["best_accuracy"])
    print(f"\nBest sampler: {best_sampler[0]} with {best_sampler[1]['best_accuracy']:.4f}")

    random_best = results.get("RandomSampler", {}).get("best_accuracy", 0)
    tpe_best = results.get("TPESampler", {}).get("best_accuracy", 0)
    if tpe_best > random_best:
        print(f"TPESampler improvement over RandomSampler: {(tpe_best - random_best) / random_best * 100:.2f}%")
    elif random_best > tpe_best:
        print(f"RandomSampler performed better by: {(random_best - tpe_best) / tpe_best * 100:.2f}%")
    else:
        print("Both samplers achieved the same best accuracy")

    # Convergence analysis
    for name, result in results.items():
        early_improvement = result["best_so_far"][4] - result["best_so_far"][0]
        late_improvement = result["best_so_far"][-1] - result["best_so_far"][-5]
        direction = "Converged quickly" if early_improvement > late_improvement else "Continued improving"
        print(f"{name}: early +{early_improvement:.4f}, late +{late_improvement:.4f} -> {direction}")

    # Save results
    final_results = {
        "n_trials_per_sampler": N_TRIALS_PER_SAMPLER,
        "total_trials": N_TRIALS_PER_SAMPLER * len(samplers),
        "quantization_types": [cls.__name__ for cls in QUANTIZATION_TYPES],
        "samplers": results,
    }
    results_file = output_dir / 'sampler_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to '{results_file}'")
