# Tutorial 6: Mixed Precision Quantization Search with Mase and Optuna

import dill
import optuna
import torch
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from optuna.samplers import RandomSampler
from transformers import AutoModelForSequenceClassification

from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60, flush=True)
print("Tutorial 6: Mixed Precision Search with Mase + Optuna", flush=True)
print("=" * 60, flush=True)

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# ── Step 1: Load base model ────────────────────────────────────────────────────
print("\n[1/4] Loading base model...", flush=True)

# Option A: load from Tutorial 5 NAS checkpoint
with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f)
print("      Loaded from tutorial_5_best_model.pkl  ✓", flush=True)

# Option B: load from HuggingFace directly (use if Tutorial 5 checkpoint is not available)
# base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# print(f"      Loaded {checkpoint} from HuggingFace  ✓", flush=True)

# ── Step 2: Load dataset ───────────────────────────────────────────────────────
print("\n[2/4] Loading and tokenizing IMDb dataset...", flush=True)
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
print(f"      Dataset loaded: {len(dataset['train'])} train / {len(dataset['test'])} test", flush=True)

# ── Step 3: Define search space & model constructor ───────────────────────────
print("\n[3/4] Defining mixed-precision search space...", flush=True)
search_space = {
    "linear_layer_choices": [torch.nn.Linear, LinearInteger],
}
print(f"      Choices per linear layer: {[c.__name__ for c in search_space['linear_layer_choices']]}", flush=True)


def construct_model(trial):
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue

        new_layer_cls = trial.suggest_categorical(
            f"{name}_type",
            search_space["linear_layer_choices"],
        )

        if new_layer_cls == torch.nn.Linear:
            continue

        kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}

        if new_layer_cls == LinearInteger:
            kwargs["config"] = {
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "weight_width": 8,
                "weight_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
            }

        new_layer = new_layer_cls(**kwargs)
        new_layer.weight.data = layer.weight.data
        deepsetattr(trial_model, name, new_layer)

    return trial_model


N_TRIALS = 1
_trial_bar = None
_trial_count = [0]


def objective(trial):
    _trial_count[0] += 1
    if _trial_bar is not None:
        _trial_bar.set_description(f"Trial {_trial_count[0]}/{N_TRIALS}")

    model = construct_model(trial)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()
    result = trainer.evaluate()
    acc = result["eval_accuracy"]

    if _trial_bar is not None:
        _trial_bar.set_postfix(accuracy=f"{acc:.4f}")
        _trial_bar.update(1)

    trial.set_user_attr("model", model)
    return acc


# ── Step 4: Launch mixed-precision search ─────────────────────────────────────
print(f"\n[4/4] Launching mixed-precision search ({N_TRIALS} trial(s))...", flush=True)
sampler = RandomSampler()

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-mixed-precision-study",
    sampler=sampler,
)

with tqdm(total=N_TRIALS, desc="MP search trials", unit="trial") as bar:
    _trial_bar = bar
    study.optimize(objective, n_trials=N_TRIALS, timeout=60 * 60 * 24)

print(f"\n      Best accuracy: {study.best_trial.value:.4f}", flush=True)

quantized = sum(
    1 for k, v in study.best_trial.params.items()
    if v != torch.nn.Linear and "type" in k
)
total_linear = sum(
    1 for k in study.best_trial.params if "type" in k
)
print(f"      Quantized layers: {quantized}/{total_linear}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 6 complete!", flush=True)
print("=" * 60, flush=True)
