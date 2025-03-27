import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from copy import deepcopy
import logging
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

from pathlib import Path
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.graph.transforms.pruning import MovementTrackingCallback
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
    calculate_avg_bits_mg_analysis_pass,
)
from datasets import load_dataset
from chop.models import DataCollatorCTCWithPadding
from chop.dataset import MaseDataModule
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Define the base model, tokenizer, and dataset
# ------------------------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

# Get the dataset & tokenizer
tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    tokenizer_checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

base_model = AutoModelForCTC.from_pretrained(checkpoint).eval()
base_encoder = base_model.wav2vec2
base_ctc_head = base_model.lm_head

batch_size = 2
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=checkpoint,
    num_workers=0,
    processor=processor
)
data_module.setup()

# ------------------------------------------------
# 2. Prepare the base MaseGraph (CPU by default)
# ------------------------------------------------

base_model = base_model.cpu()  # keep everything on CPU initially

mg = MaseGraph(
    base_encoder,
    hf_input_names=["input_values", "attention_mask"],
)
mg, _ = init_metadata_analysis_pass(mg)

dummy_in = {
    "input_values": torch.zeros((1, 16000), dtype=torch.float32),
    "attention_mask": torch.ones((1, 16000), dtype=torch.long),
}
mg, _ = add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_in,
        "add_value": True,
        "force_device_meta": False,
    }
)

# ------------------------------------------------
# 3. Define config dicts for SmoothQuant & quant.
# ------------------------------------------------

# We will vary alpha for SmoothQuant
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# We repeat the entire alpha sweep multiple times to see variance
num_runs = 3

# We'll gather results from all runs here
all_results = []

# ------------------------------------------------
# 4. Main experiment loop
# ------------------------------------------------
for run_index in range(num_runs):
    print(f"\n==== Starting RUN {run_index+1} of {num_runs} ====")

    for alpha in alpha_values:
        print(f"\n===== Starting experiment with alpha={alpha} (Run {run_index+1}) =====")

        # 4.1 Make a fresh copy of the base model on CPU
        model_copy = deepcopy(base_model).eval().cpu()
        encoder_copy = model_copy.wav2vec2

        # -------------------------------
        # A) Quantization Only (no SQ)
        # -------------------------------
        mg_no_sq = MaseGraph(encoder_copy, hf_input_names=["input_values", "attention_mask"])
        mg_no_sq, _ = init_metadata_analysis_pass(mg_no_sq)
        mg_no_sq, _ = add_common_metadata_analysis_pass(
            mg_no_sq,
            pass_args={
                "dummy_in": dummy_in,
                "add_value": True,
                "force_device_meta": False,
            }
        )

        # Check bits before transformations (purely for reference)
        _, avg_bits_before_no_sq = calculate_avg_bits_mg_analysis_pass(mg_no_sq)
        data_bits_before_no_sq = avg_bits_before_no_sq["data_avg_bit"]
        weight_bits_before_no_sq = avg_bits_before_no_sq["w_avg_bit"]

        # Define the config for standard quant (16-bit integer)
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
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
            "conv1d": {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
            "conv2d": {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
        }
        

        # Skip the SmoothQuant pass entirely (Quant-Only scenario)
        mg_no_sq, _ = quantize_transform_pass(mg_no_sq, pass_args=quantization_config)

        # Check bits after quant
        _, avg_bits_quant_no_sq = calculate_avg_bits_mg_analysis_pass(mg_no_sq)
        data_bits_quant_no_sq = avg_bits_quant_no_sq["data_avg_bit"]
        weight_bits_quant_no_sq = avg_bits_quant_no_sq["w_avg_bit"]

        # Measure runtime metrics
        runtime_analysis_config = {
            "num_batches": 35,
            "num_GPU_warmup_batches": 2,
            "test": True,
            "data_module": data_module,
            "model": checkpoint,
            "accelerator": "cuda",
            "task": "ctc",
            "decoder": decoder,
            "beam_width": 10,
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            "sample_rate": 16000,
            "ctc_head": model_copy.lm_head,  # Make sure to use the same ctc_head reference
        }

        _, results_no_sq = runtime_analysis_pass(mg_no_sq, pass_args=runtime_analysis_config)
        wer_no_sq = results_no_sq["Average WER"]
        latency_ms_no_sq = results_no_sq["Average Latency"]
        rtf_no_sq = results_no_sq["Average RTF"]
        avg_gpu_power_no_sq = results_no_sq["Average GPU Power Usage"]
        energy_consumption_no_sq = results_no_sq["Inference Energy Consumption"]

        print(f"   [Alpha={alpha} - NO SQ] WER={wer_no_sq:.5g}, Latency={latency_ms_no_sq:.5g} ms, "
              f"RTF={rtf_no_sq:.5g}, GPU Power={avg_gpu_power_no_sq:.5g} W, "
              f"Energy={energy_consumption_no_sq:.5g} mWh")

        all_results.append({
            "run_index": run_index,
            "alpha": alpha,
            "method": "Quant_Only",
            "data_bits_before": data_bits_before_no_sq,
            "weight_bits_before": weight_bits_before_no_sq,
            "data_bits_smooth": None,   # Not applicable in no-SQ scenario
            "weight_bits_smooth": None, # Not applicable in no-SQ scenario
            "data_bits_quant": data_bits_quant_no_sq,
            "weight_bits_quant": weight_bits_quant_no_sq,
            "WER": wer_no_sq,
            "Latency_ms": latency_ms_no_sq,
            "RTF": rtf_no_sq,
            "GPU_Power_W": avg_gpu_power_no_sq,
            "Energy_mWh": energy_consumption_no_sq,
        })

        # -------------------------------
        # B) SmoothQuant + Quantization
        # -------------------------------
        # Make another fresh copy so we haven't modified the original
        model_copy_sq = deepcopy(base_model).eval().cpu()
        encoder_copy_sq = model_copy_sq.wav2vec2

        mg_sq = MaseGraph(encoder_copy_sq, hf_input_names=["input_values", "attention_mask"])
        mg_sq, _ = init_metadata_analysis_pass(mg_sq)
        mg_sq, _ = add_common_metadata_analysis_pass(
            mg_sq,
            pass_args={
                "dummy_in": dummy_in,
                "add_value": True,
                "force_device_meta": False,
            }
        )

        # Check bits before transformations
        _, avg_bits_before_sq = calculate_avg_bits_mg_analysis_pass(mg_sq)
        data_bits_before_sq = avg_bits_before_sq["data_avg_bit"]
        weight_bits_before_sq = avg_bits_before_sq["w_avg_bit"]

        # SmoothQuant pass
        smoothquant_config = {
            "smoothquant": True,
            "alpha": alpha,
            "model": checkpoint,
            "task": "ctc",
            "dataset": dataset_name,
            "accelerator": "cuda",
            "data_module": data_module,
            "batch_size": batch_size,
        }

        

        mg_sq, _ = onnx_runtime_interface_pass(mg_sq, pass_args=smoothquant_config)

        # Bits after SmoothQuant, before quant
        _, avg_bits_smooth_sq = calculate_avg_bits_mg_analysis_pass(mg_sq)
        data_bits_smooth_sq = avg_bits_smooth_sq["data_avg_bit"]
        weight_bits_smooth_sq = avg_bits_smooth_sq["w_avg_bit"]

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
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
            "conv1d": {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
            "conv2d": {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
        }

        # Apply standard quantization
        mg_sq, _ = quantize_transform_pass(mg_sq, pass_args=quantization_config)

        # Bits after quant
        _, avg_bits_quant_sq = calculate_avg_bits_mg_analysis_pass(mg_sq)
        data_bits_quant_sq = avg_bits_quant_sq["data_avg_bit"]
        weight_bits_quant_sq = avg_bits_quant_sq["w_avg_bit"]

        # Measure runtime metrics
        runtime_analysis_config_sq = {
            "num_batches": 35,
            "num_GPU_warmup_batches": 2,
            "test": True,
            "data_module": data_module,
            "model": checkpoint,
            "accelerator": "cuda",
            "task": "ctc",
            "decoder": decoder,
            "beam_width": 10,
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            "sample_rate": 16000,
            "ctc_head": model_copy_sq.lm_head,
        }

        _, results_sq = runtime_analysis_pass(mg_sq, pass_args=runtime_analysis_config_sq)
        wer_sq = results_sq["Average WER"]
        latency_ms_sq = results_sq["Average Latency"]
        rtf_sq = results_sq["Average RTF"]
        avg_gpu_power_sq = results_sq["Average GPU Power Usage"]
        energy_consumption_sq = results_sq["Inference Energy Consumption"]

        print(f"   [Alpha={alpha} - SQ+Quant] WER={wer_sq:.5g}, Latency={latency_ms_sq:.5g} ms, "
              f"RTF={rtf_sq:.5g}, GPU Power={avg_gpu_power_sq:.5g} W, "
              f"Energy={energy_consumption_sq:.5g} mWh")

        all_results.append({
            "run_index": run_index,
            "alpha": alpha,
            "method": "SQ_Quant",
            "data_bits_before": data_bits_before_sq,
            "weight_bits_before": weight_bits_before_sq,
            "data_bits_smooth": data_bits_smooth_sq,
            "weight_bits_smooth": weight_bits_smooth_sq,
            "data_bits_quant": data_bits_quant_sq,
            "weight_bits_quant": weight_bits_quant_sq,
            "WER": wer_sq,
            "Latency_ms": latency_ms_sq,
            "RTF": rtf_sq,
            "GPU_Power_W": avg_gpu_power_sq,
            "Energy_mWh": energy_consumption_sq,
        })

# ------------------------------------------------
# 5. Save ALL runs to CSV
# ------------------------------------------------
df_all = pd.DataFrame(all_results)
df_all.to_csv("smoothquant_experiments_all_runs.csv", index=False)
print("\nSaved results to smoothquant_experiments_all_runs.csv")
print(df_all.head())

# ------------------------------------------------
# 6. Aggregate results (mean & std) and save
#    (group by both alpha and method)
# ------------------------------------------------
agg_df = df_all.groupby(["alpha", "method"]).agg(
    WER_mean=("WER","mean"),
    WER_std=("WER","std"),
    Latency_mean=("Latency_ms","mean"),
    Latency_std=("Latency_ms","std"),
    RTF_mean=("RTF","mean"),
    RTF_std=("RTF","std"),
    GPU_Power_mean=("GPU_Power_W","mean"),
    GPU_Power_std=("GPU_Power_W","std"),
    Energy_mean=("Energy_mWh","mean"),
    Energy_std=("Energy_mWh","std"),
    data_before_mean=("data_bits_before","mean"),
    data_before_std=("data_bits_before","std"),
    data_smooth_mean=("data_bits_smooth","mean"),
    data_smooth_std=("data_bits_smooth","std"),
    data_quant_mean=("data_bits_quant","mean"),
    data_quant_std=("data_bits_quant","std"),
    weight_before_mean=("weight_bits_before","mean"),
    weight_before_std=("weight_bits_before","std"),
    weight_smooth_mean=("weight_bits_smooth","mean"),
    weight_smooth_std=("weight_bits_smooth","std"),
    weight_quant_mean=("weight_bits_quant","mean"),
    weight_quant_std=("weight_bits_quant","std"),
).reset_index()

agg_df.to_csv("smoothquant_experiments_aggregated.csv", index=False)
print("\nSaved aggregated results to smoothquant_experiments_aggregated.csv")
print(agg_df.head())

# ------------------------------------------------
# 7. Plot: For each alpha, we show 2 lines
#    (Quant_Only vs SQ_Quant) for all metrics
# ------------------------------------------------

fig, axs = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle("Comparison: Quant-Only vs SmoothQuant+Quant", fontsize=16)

# Helper function to plot each metric with 2 lines
def plot_metric(ax, metric_name, y_label):
    """
    metric_name: e.g. 'WER' or 'Latency_ms'
    y_label: label for the y-axis
    """
    for method in ["Quant_Only", "SQ_Quant"]:
        sel = agg_df[agg_df["method"] == method]
        x = sel["alpha"]
        y_mean = sel[f"{metric_name}_mean"]
        y_std = sel[f"{metric_name}_std"]
        ax.errorbar(
            x, y_mean, yerr=y_std, marker='o', capsize=3, label=method
        )
    ax.set_title(f"{metric_name} vs Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel(y_label)
    ax.legend()

# (0,0): WER
plot_metric(axs[0, 0], "WER", "WER")

# (0,1): Latency
plot_metric(axs[0, 1], "Latency", "Latency (ms)")

# (1,0): RTF
plot_metric(axs[1, 0], "RTF", "Real-Time Factor")

# (1,1): GPU Power
plot_metric(axs[1, 1], "GPU_Power", "GPU Power (W)")

# (2,0): Energy
plot_metric(axs[2, 0], "Energy", "Energy (mWh)")

# (2,1): Data Bits after quant (comparing final data bits)
for method in ["Quant_Only", "SQ_Quant"]:
    sel = agg_df[agg_df["method"] == method]
    x = sel["alpha"]
    y_mean = sel["data_quant_mean"]
    y_std = sel["data_quant_std"]
    axs[2, 1].errorbar(
        x, y_mean, yerr=y_std, marker='o', capsize=3, label=method
    )
axs[2, 1].set_title("Data Bits (After Quant) vs Alpha")
axs[2, 1].set_xlabel("Alpha")
axs[2, 1].set_ylabel("Data Bits")
axs[2, 1].legend()

# (3,0): Weight Bits after quant (comparing final weight bits)
for method in ["Quant_Only", "SQ_Quant"]:
    sel = agg_df[agg_df["method"] == method]
    x = sel["alpha"]
    y_mean = sel["weight_quant_mean"]
    y_std = sel["weight_quant_std"]
    axs[3, 0].errorbar(
        x, y_mean, yerr=y_std, marker='o', capsize=3, label=method
    )
axs[3, 0].set_title("Weight Bits (After Quant) vs Alpha")
axs[3, 0].set_xlabel("Alpha")
axs[3, 0].set_ylabel("Weight Bits")
axs[3, 0].legend()

# (3,1): hide or set to off
axs[3, 1].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("smoothquant_experiments_plots.png")
plt.show()

print("\nDone! Check 'smoothquant_experiments_all_runs.csv' and 'smoothquant_experiments_aggregated.csv' for numeric results, and 'smoothquant_experiments_plots.png' for plots.")
