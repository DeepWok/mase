# Tutorial 5: Neural Architecture Search (NAS) with Mase and Optuna

import dill
import optuna
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification
from optuna.samplers import RandomSampler

from chop import MaseGraph
from chop.nn.modules import Identity
from chop.pipelines import CompressionPipeline
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60, flush=True)
print("Tutorial 5: NAS with Mase + Optuna (BERT / IMDb)", flush=True)
print("=" * 60, flush=True)

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# ── Step 1: Load dataset ───────────────────────────────────────────────────────
print("\n[1/5] Loading and tokenizing IMDb dataset...", flush=True)
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
print(f"      Dataset loaded: {len(dataset['train'])} train / {len(dataset['test'])} test", flush=True)

# ── Step 2: Define search space ───────────────────────────────────────────────
print("\n[2/5] Defining NAS search space...", flush=True)
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [nn.Linear, Identity],
}
for k, v in search_space.items():
    print(f"      {k}: {v}", flush=True)


# ── Step 3: Model constructor + objective ─────────────────────────────────────
def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][idx])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_cls = trial.suggest_categorical(
                f"{name}_type", search_space["linear_layer_choices"]
            )
            if new_cls == Identity:
                deepsetattr(trial_model, name, Identity())
    return trial_model


_trial_bar = None
_trial_count = [0]
N_TRIALS = 1


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


# ── Step 4: Launch NAS search ─────────────────────────────────────────────────
print(f"\n[3/5] Launching NAS search ({N_TRIALS} trial(s))...", flush=True)
sampler = RandomSampler()
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

with tqdm(total=N_TRIALS, desc="NAS trials", unit="trial") as bar:
    _trial_bar = bar
    study.optimize(objective, n_trials=N_TRIALS, timeout=60 * 60 * 24)

print(f"\n      Best accuracy: {study.best_trial.value:.4f}", flush=True)
print(f"      Best params:   {study.best_trial.params}", flush=True)

# ── Step 5: Save best model ────────────────────────────────────────────────────
print("\n[4/5] Saving best model...", flush=True)
best_model = study.best_trial.user_attrs["model"].cpu()
save_path = f"{Path.home()}/tutorial_5_best_model.pkl"
with open(save_path, "wb") as f:
    dill.dump(best_model, f)
print(f"      Saved to {save_path}", flush=True)

# ── Step 6: Compress with CompressionPipeline ─────────────────────────────────
print("\n[5/5] Compressing best model (quantize + prune)...", flush=True)
mg = MaseGraph(best_model)
pipe = CompressionPipeline()

quantization_config = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

pruning_config = {
    "weight": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
    "activation": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
}

mg, _ = pipe(
    mg,
    pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    },
)
print("      CompressionPipeline  ✓", flush=True)

mg.export(f"{Path.home()}/tutorial_5_nas_compressed", save_format="state_dict")
print(f"      Compressed model saved to {Path.home()}/tutorial_5_nas_compressed", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 5 complete!", flush=True)
print("=" * 60, flush=True)
