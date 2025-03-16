# import random
import dill
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.tools import get_trainer, get_tokenized_dataset # type: ignore
from chop.tools.utils import deepsetattr # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from pyctcdecode import build_ctcdecoder
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)
from chop.dataset.nlp.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
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
)

baseline_metrics = None

# -------------------------------
# 1. Define the model and dataset
# -------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "librispeech_asr"

# Logic inside get_tockenized_dataset needs to be improved using nyal's changes
dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

# Logic needs to be improved for seperated train and test split
tokenized_dataset = DatasetDict({
    "train": dataset,
    "test": dataset
})

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

model = AutoModelForCTC.from_pretrained(checkpoint)
model.config.gradient_checkpointing = True
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head    # dynamic CTC head, separate this

# -------------------------------
# 2. Import ONNX dataset & Wrapper
# -------------------------------

dataset_path = Path("./preprocessed_data")
condensed_dataset = CondensedLibrispeechASRDataset(dataset_path=dataset_path, split="train") # Choose valid split
condensed_dataset.prepare_data()
condensed_dataset.setup()

data_module = MaseDataModule(
    name="nyalpatel/condensed_librispeech_asr",
    batch_size=1,
    model_name=checkpoint,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

class ONNXWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder 

    def forward(self, inputs):
        if isinstance(inputs, dict):
            input_values = inputs["input_values"]
            attention_mask = inputs["attention_mask"]
        else:
            input_values = inputs
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
        return self.encoder(input_values, attention_mask=attention_mask)
    
    @property
    def graph(self):
        # Expose the underlying FX graph for later passes
        return self.encoder.graph

# -------------------------------
# 3. Define the MASE graph & metadata
# -------------------------------

mg = MaseGraph(
    encoder,
    hf_input_names=[
        "input_values",
        "attention_mask",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)

dummy_in = {
    "input_values": torch.zeros((1, 16000), dtype=torch.float32),
    "attention_mask": torch.ones((1, 16000), dtype=torch.long),
}

mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_in,
        "add_value": True,
        "force_device_meta": False,
    }
)

combined_model = CombinedWav2Vec2CTC(
        encoder=mg.model,
        ctc_head=ctc_head, 
        decoder=decoder,
        beam_width=10
    )

# -------------------------------
# 4. Define Search Space & Baseline
# -------------------------------

def run_onnx_baseline(mg, data_module, checkpoint, dataset_name, decoder, tokenizer):
    """
    Performs an ONNX pass without quantization or SmoothQuant,
    then runs runtime analysis to get baseline metrics.
    """
    mg.model = ONNXWrapper(mg.model)

    # alpha=0 => no smoothing
    smoothquant_config = {
        "smoothquant": True,
        "alpha": 0.0,
        "model": checkpoint,
        "task": "ctc",
        "dataset": dataset_name,
        "accelerator": "cuda",
        "data_module": data_module,
        "batch_size": 1,
    }

    runtime_analysis_config = {
        "num_batches": 100,
        "num_GPU_warmup_batches": 5,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": 2,
        "sample_rate": 16000,
    }

    mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
    _, baseline_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
    
    print("\nBaseline (no quant, alpha=0) ONNX Metrics:")
    for k, v in baseline_results.items():
        print(f"  {k}: {v}")
    return baseline_results

precision_choices = [
    # nn.Linear,  
    LinearInteger,
    # LinearMinifloatDenorm,
    # LinearMinifloatIEEE,
    # LinearLog,
    # LinearBlockFP,
    # LinearBlockLog,
    # LinearBinary,
    # LinearBinaryScaling,
]

width_choices = [16, 32]
frac_width_choices = [8, 16]

# -------------------------------
# 5. Construct Quantized Model
# -------------------------------

def construct_quantized_model(trial, chosen_precision):
    """
    Constructs a quantized model where nn.Linear layers in the encoder
    are replaced with the chosen quantization precision type.
    """
    # Copy the base encoder to avoid modifying the original
    encoder_quant = deepcopy(mg.model)

    for name, layer in encoder_quant.named_modules():
        if isinstance(layer, nn.Linear):
            if chosen_precision == nn.Linear:
                continue

            kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}

            # [!!! FIX] If chosen_precision is LinearInteger or others:
            if chosen_precision == LinearInteger:
                config = {
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", width_choices),
                    "weight_frac_width": trial.suggest_categorical(f"{name}_weight_frac_width", frac_width_choices),
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", width_choices),
                    "data_in_frac_width": trial.suggest_categorical(f"{name}_data_in_frac_width", frac_width_choices),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", width_choices),
                    "bias_frac_width": trial.suggest_categorical(f"{name}_bias_frac_width", frac_width_choices),
                    "floor": False,
                }
            elif chosen_precision in [LinearMinifloatDenorm, LinearMinifloatIEEE]:
                config = {
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", width_choices),
                    "weight_exponent_width": 5,
                    "weight_exponent_bias": 15,
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", width_choices),
                    "data_in_exponent_width": 5,
                    "data_in_exponent_bias": 15,
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", width_choices),
                    "bias_exponent_width": 5,
                    "bias_exponent_bias": 15,
                }
            else:
                config = {}

            new_layer = chosen_precision(**kwargs, config=config)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()

            deepsetattr(encoder_quant, name, new_layer)

    # Create the combined model with the quantized encoder
    quantized_model = CombinedWav2Vec2CTC(
        encoder=encoder_quant,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    return quantized_model

# -------------------------------
# 6. Objective Function for Optimization
# -------------------------------

def objective(trial, chosen_precision, baseline_metrics):
    """
    Mixed-precision search objective.
    1) Quantize
    2) (Optional) train quantized model
    3) ONNX pass (with alpha in [0.0,1.0])
    4) Compare final ONNX metrics to baseline
    """
    # 1) Build the quantized model
    quant_model = construct_quantized_model(trial, chosen_precision)
    
    # 2) Train the quantized model
    trainer = get_trainer(
        model=quant_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        data_collator=data_collator,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    trainer.train()
    
    eval_results = trainer.evaluate()
    wer = eval_results.get("eval_wer", None)
    loss = eval_results.get("eval_loss", None)
    trial.set_user_attr("eval_wer_pt", wer)
    trial.set_user_attr("eval_loss_pt", loss)

    # 3) SmoothQuant alpha from search space
    alpha_val = trial.suggest_float("smoothquant_alpha", 0.0, 1.0, step=0.1)

    mg.model = trainer.model.encoder
    mg.model = ONNXWrapper(mg.model)

    smoothquant_config = {
        "smoothquant": True,
        "alpha": alpha_val,
        "model": checkpoint,
        "task": "ctc",
        "dataset": dataset_name,
        "accelerator": "cuda",
        "data_module": data_module,
        "batch_size": 1,
    }

    runtime_analysis_config = {
        "num_batches": 100,
        "num_GPU_warmup_batches": 5,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": 2,
        "sample_rate": 16000,
    }

    mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
    _, onnx_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
    trial.set_user_attr("onnx_metrics", onnx_results)

    # 4) Compare final ONNX metrics to baseline
    relevant_keys = [
        "Average WER",
        "Average Latency",
        "Average RTF",
        "Average GPU Power Usage",
        "Inference Energy Consumption",
    ]

    for k in relevant_keys:
        val = onnx_results.get(k, float("inf"))
        trial.set_user_attr(f"onnx_{k}", val)

    # For WER, latency, RTF, etc. smaller=better => ratio = trial_val / baseline_val
    ratios = []
    for k in relevant_keys:
        trial_val = onnx_results.get(k, None)
        base_val = baseline_metrics.get(k, None)
        if trial_val is not None and base_val is not None and base_val != 0:
            ratio = trial_val / base_val
            ratios.append(ratio)

    composite_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    composite_metric = 1.0 - composite_ratio
    trial.set_user_attr("composite_ratio", composite_ratio)
    trial.set_user_attr("composite_metric", composite_metric)

    return composite_metric

# -------------------------------
# 7. Run Optimization Studies
# -------------------------------

def run_study_for_precision(chosen_precision, n_trials=5):
    """
    Runs an Optuna study for a fixed precision type, comparing results to the baseline.
    """
    global baseline_metrics
    if baseline_metrics is None:
        print("Running baseline ONNX pass (no quant, alpha=0) for reference...")
        baseline_metrics = run_onnx_baseline(mg, data_module, checkpoint, dataset_name, decoder, tokenizer)

    sampler = TPESampler()
    study_name = f"study_{chosen_precision.__name__}" if chosen_precision != nn.Linear else "study_FullPrecision"
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

    study.optimize(lambda trial: objective(trial, chosen_precision, baseline_metrics),
                   n_trials=n_trials,
                   timeout=60 * 60 * 24)

    best_trial = study.best_trial
    print(f"\nBest trial for {chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'}:")
    print("  Trial Number:", best_trial.number)
    print("  Best Composite Metric:", best_trial.value)
    print("  PT-based WER:", best_trial.user_attrs.get("eval_wer_pt", "N/A"))
    print("  ONNX-based metrics:", best_trial.user_attrs.get("onnx_metrics", {}))

    results = []
    for t in sorted(study.trials, key=lambda t: t.number):
        composite = t.value
        onnx_m = t.user_attrs.get("onnx_metrics", {})
        alpha_val = t.params.get("smoothquant_alpha", None)

        row = {
            "trial_number": t.number,
            "trial_composite_metric": composite,
            "smoothquant_alpha": alpha_val,
            "onnx_wer": onnx_m.get("Average WER", None),
            "onnx_latency": onnx_m.get("Average Latency", None),
            "onnx_rtf": onnx_m.get("Average RTF", None),
            "onnx_power": onnx_m.get("Average GPU Power Usage", None),
            "onnx_energy": onnx_m.get("Inference Energy Consumption", None),
        }
        results.append(row)

    df = pd.DataFrame(results)
    csv_name = f"optuna_results_{chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Results for precision {chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'} saved to {csv_name}")

    return df, best_trial

# -------------------------------
# 8. Get Best Model and Save Results
# -------------------------------

def save_best_model(best_trial, chosen_precision):
    best_model = construct_quantized_model(best_trial, chosen_precision)
    with open(f"best_model_{chosen_precision.__name__}.pkl", "wb") as f:
        dill.dump(best_model, f)
    print(f"Best model saved to best_model_{chosen_precision.__name__}.pkl")
    return best_model

# -------------------------------
# 9. Main execution
# -------------------------------

all_results = []
for prec in precision_choices:
    print(f"\nRunning study for precision type: {prec.__name__ if prec != nn.Linear else 'FullPrecision'}")
    df_prec, best_trial = run_study_for_precision(prec, n_trials=10)

    # Save the best model for this precision type
    best_model = save_best_model(best_trial, prec)

    # Append results
    df_prec["precision_type"] = prec.__name__ if prec != nn.Linear else "FullPrecision"
    all_results.append(df_prec)

    # If you want a WER progress plot, you must track PT-based WER or ONNX-based WER over trials.
    # For example, let's track "PT-based WER" from user_attrs:

    if "trial_wer" not in df_prec.columns:
        # Let's add a column for trial_wer from user attrs "eval_wer_pt"
        df_prec["trial_wer"] = df_prec.apply(
            lambda row: [t.user_attrs.get("eval_wer_pt", None) for t in best_trial.study.trials
                         if t.number == row["trial_number"]][0],
            axis=1
        )

    df_temp = df_prec.sort_values("trial_number")
    trial_nums = df_temp["trial_number"].tolist()
    wer_values = df_temp["trial_wer"].tolist()
    cum_best_wer = []
    current_best_wer = float("inf")
    for val in wer_values:
        if val is not None:
            current_best_wer = min(current_best_wer, val)
        cum_best_wer.append(current_best_wer)

    plt.figure(figsize=(6,4))
    plt.plot(trial_nums, cum_best_wer, marker="o",
             label=prec.__name__ if prec != nn.Linear else "FullPrecision")
    plt.xlabel("Trial Number")
    plt.ylabel("Cumulative Best PT-based WER")
    plt.title(f"PT WER Progress for {prec.__name__ if prec != nn.Linear else 'FullPrecision'}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"optuna_wer_progress_{prec.__name__ if prec != nn.Linear else 'FullPrecision'}.png")
    plt.close()

combined_df = pd.concat(all_results, ignore_index=True)

def compute_percentage_change(current, baseline):
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline * 100.0

for metric_col, metric_key in [
    ("onnx_wer", "Average WER"),
    ("onnx_latency", "Average Latency"),
    ("onnx_rtf", "Average RTF"),
    ("onnx_power", "Average GPU Power Usage"),
    ("onnx_energy", "Inference Energy Consumption"),
]:
    baseline_val = baseline_metrics.get(metric_key, None)
    combined_df[f"{metric_col}_pct_change"] = combined_df[metric_col].apply(
        lambda x: compute_percentage_change(x, baseline_val) if x is not None and baseline_val else None
    )

# Example plot: WER % change vs baseline
plt.figure(figsize=(10,6))
for precision, group in combined_df.groupby("precision_type"):
    group = group.sort_values("trial_number")
    wer_pct_change = group["onnx_wer_pct_change"].tolist()
    trial_nums = group["trial_number"].tolist()

    plt.plot(trial_nums, wer_pct_change, marker="o", label=precision)

plt.xlabel("Trial Number")
plt.ylabel("WER % Change vs Baseline")
plt.title("WER Improvement Over Baseline by Precision Type")
plt.legend()
plt.grid(True)
plt.savefig("optuna_combined_wer_pct_change.png")
plt.close()

# Save combined results
combined_df.to_csv("optuna_combined_results.csv", index=False)
print("Combined results saved to optuna_combined_results.csv")
