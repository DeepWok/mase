import random
import dill
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler

from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.tools import get_trainer # type: ignore
from chop.tools.utils import deepsetattr # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding
from pyctcdecode import build_ctcdecoder
from chop.passes.graph.analysis.quantization.calculate_avg_bits import calculate_avg_bits_mg_analysis_pass

# Import supported linear quantizers
from torch import nn
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

# -------------------------------
# 1. Define the model and dataset
# -------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
tokenizer = processor.tokenizer

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

dataset = load_dataset(dataset_name, split="test.clean")
sample_list = list(dataset.take(100))

def preprocess_function(example):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    inputs = processor(audio=audio_array, sampling_rate=int(sampling_rate), return_tensors="pt", padding=True)
    attention_mask = torch.ones(inputs.input_values.shape, dtype=torch.long)

    with processor.as_target_processor():
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids

    return {
        "input_values": inputs.input_values.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0)
    }

small_dataset = Dataset.from_list(sample_list)
small_dataset = small_dataset.map(
    preprocess_function,
    remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
)

# Convert to DatasetDict (as required by MASE)
tokenized_dataset = DatasetDict({
    "train": small_dataset,
    "test": small_dataset
})


model = AutoModelForCTC.from_pretrained(checkpoint)
model.config.gradient_checkpointing = True
base_encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head     # dynamic CTC head, separate this

# -------------------------------
# 2. Define the MASE graph
# -------------------------------

mg = MaseGraph(
    base_encoder,
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

# -------------------------------
# 3. Define the Combined Model
# -------------------------------

class CombinedWav2Vec2CTC(nn.Module):
    def __init__(self, encoder, ctc_head, blank_id=0, beam_width=10, decoder=None):
        """
        Args:
            encoder: The traced encoder (e.g., mg.model)
            ctc_head: The CTC head (usually a linear layer)
            blank_id: The token ID for the blank symbol (typically 0)
            beam_width: Width for beam search decoding (if using a decoder)
            decoder: (Optional) A beam search decoder (e.g., from pyctcdecode)
        """
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.decoder = decoder  # Only used during inference

    def forward(self, input_values, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = encoder_out["last_hidden_state"]
        logits = self.ctc_head(hidden_states) # outputs tensor as expected

        output = {"logits": logits, "labels": labels}

        if labels is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            batch_size, time_steps, _ = logits.shape
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
            target_lengths = (labels != -100).sum(dim=1)

            loss = F.ctc_loss(log_probs, labels, input_lengths, target_lengths, blank=self.blank_id, reduction="mean", zero_infinity=True)
            output["loss"] = loss
        else:
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                output["transcription"] = transcription
        return output

# -------------------------------
# 4. Quantization Search Space Definition
# -------------------------------

# Define the list of precision types to test
precision_choices = [
    #nn.Linear,  
    LinearInteger,
    # LinearMinifloatDenorm,
    # LinearMinifloatIEEE,
    # LinearLog,
    # LinearBlockFP,
    # LinearBlockLog,
    # LinearBinary,
    # LinearBinaryScaling,
]

# Allowed choices for widths and fractional widths
width_choices = [16, 32]
frac_width_choices = [8, 16]

def construct_quantized_model(trial, chosen_precision):
    """
    Constructs a quantized model where nn.Linear layers in the encoder
    are replaced with the chosen quantization precision type.
    """
    # Copy the base encoder to avoid modifying the original
    encoder = deepcopy(mg.model)

    for name, layer in encoder.named_modules():
        if isinstance(layer, nn.Linear):
            # Skip quantization for full precision
            if chosen_precision == nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # Depending on the chosen precision, create a configuration
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
            # Add other precision configurations as needed
            else:
                config = {}

            # Create the new quantized layer
            new_layer = chosen_precision(**kwargs, config=config)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()

            # Replace the original layer with the quantized one
            deepsetattr(encoder, name, new_layer)

    # Create the combined model with the quantized encoder
    combined_model = CombinedWav2Vec2CTC(
        encoder=encoder,
        ctc_head=ctc_head,  # Keep the CTC head as is (not quantized)
        decoder=decoder,
        beam_width=10
    )

    return combined_model

# -------------------------------
# 5. Objective Function for Optimization
# -------------------------------

def build_graph_from_model(model):
    """Create a MaseGraph from the model for bit-width analysis"""
    encoder = model.encoder  # Extract just the encoder part

    mg = MaseGraph(
        encoder,
        hf_input_names=[
            "input_values",
            "attention_mask",
        ],
    )

    # Run the metadata passes
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

    return mg

def objective(trial, chosen_precision):
    """
    Objective function for Optuna optimization.
    Returns a composite metric balancing WER and bit-width.
    """
    # Construct the quantized model
    model = construct_quantized_model(trial, chosen_precision)

    # Get a trainer for evaluation
    trainer = get_trainer(
        model=model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",  # Word Error Rate for ASR
        num_train_epochs=1,
        data_collator=data_collator,
    )

    # Train briefly to adapt to quantization
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    wer = eval_results["eval_wer"]

    # Store the WER as a user attribute
    trial.set_user_attr("eval_wer", wer)

    # Move model to CPU for bit-width analysis
    #model = model.cpu()

    # Calculate average bit width
    #graph = build_graph_from_model(model)
    #graph, avg_bit_dict = calculate_avg_bits_mg_analysis_pass(graph, pass_args={})

    # Store the average bits results as a user attribute
    #trial.set_user_attr("avg_bit_dict", avg_bit_dict)
    #trial.set_user_attr("precision_type", chosen_precision.__name__ if chosen_precision != nn.Linear else "FullPrecision")

    #print(f"Trial {trial.number}:")
    ##print(f"WER: {wer}")
    #print("Average bits analysis:", avg_bit_dict)

    # Composite metric: Lower WER is better, so we use negative WER
    # We also want to minimize bit width, so we penalize for high bit widths
    #alpha = 0.01  # Adjust the scaling factor as needed
    #composite_metric = -wer - alpha * (avg_bit_dict['w_avg_bit'] + avg_bit_dict['data_avg_bit'])
    composite_metric = -wer

    return composite_metric

# -------------------------------
# 6. Run Optimization Studies
# -------------------------------

def run_study_for_precision(chosen_precision, n_trials=5):
    """Runs an Optuna study for a fixed precision type."""
    sampler = TPESampler()
    study_name = f"study_{chosen_precision.__name__}" if chosen_precision != nn.Linear else "study_FullPrecision"
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

    # Optimize the study with a lambda that fixes the chosen precision
    study.optimize(lambda trial: objective(trial, chosen_precision),
                  n_trials=n_trials,
                  timeout=60 * 60 * 24)  # 24-hour timeout (adjust as needed)

    # Print best trial's results
    best_trial = study.best_trial
    print(f"\nBest trial for {chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'}:")
    print("Trial Number:", best_trial.number)
    print("Best Composite Metric:", best_trial.value)
    print("Best trial's WER:", best_trial.user_attrs.get("eval_wer", {}))
    print("Best trial's avg_bit_dict:", best_trial.user_attrs.get("avg_bit_dict", {}))

    # Gather results into a DataFrame
    results = []
    current_best_composite = -float("inf")
    current_best_wer = float("inf")
    current_best_avg_bits = {"w_avg_bit": None, "data_avg_bit": None}

    for t in sorted(study.trials, key=lambda t: t.number):
        composite = t.value
        wer = t.user_attrs.get("eval_wer", None)
        if composite is not None and composite > current_best_composite:
            current_best_composite = composite
            current_best_avg_bits = t.user_attrs.get("avg_bit_dict", {"w_avg_bit": None, "data_avg_bit": None})
        if wer is not None and wer < current_best_wer:
            current_best_wer = wer

        results.append({
            "trial_number": t.number,
            "trial_composite_metric": composite,
            "trial_wer": wer,
            "precision_type": t.user_attrs.get("precision_type", "Unknown"),
            "current_best_w_avg_bit": current_best_avg_bits.get("w_avg_bit"),
            "current_best_data_avg_bit": current_best_avg_bits.get("data_avg_bit"),
            "current_best_wer": current_best_wer
        })

    df = pd.DataFrame(results)
    csv_name = f"optuna_results_{chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Results for precision {chosen_precision.__name__ if chosen_precision != nn.Linear else 'FullPrecision'} saved to {csv_name}")

    return df, best_trial

# -------------------------------
# 7. Get Best Model and Save Results
# -------------------------------

def save_best_model(best_trial, chosen_precision):
    """Recreate and save the best model from the trial"""
    best_model = construct_quantized_model(best_trial, chosen_precision)
    with open(f"best_model_{chosen_precision.__name__}.pkl", "wb") as f:
        dill.dump(best_model, f)
    print(f"Best model saved to best_model_{chosen_precision.__name__}.pkl")
    return best_model

# -------------------------------
# 8. Main execution
# -------------------------------

# Run studies for each precision type
all_results = []  # to store all results for final plotting

for prec in precision_choices:
    print(f"\nRunning study for precision type: {prec.__name__ if prec != nn.Linear else 'FullPrecision'}")
    df_prec, best_trial = run_study_for_precision(prec, n_trials=10)

    # Save the best model for this precision type
    best_model = save_best_model(best_trial, prec)

    # Append results to all_results for comparison
    df_prec["precision_type"] = prec.__name__ if prec != nn.Linear else "FullPrecision"
    all_results.append(df_prec)

    # Plot the WER progress
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
    plt.plot(trial_nums, cum_best_wer, marker="o", label=prec.__name__ if prec != nn.Linear else "FullPrecision")
    plt.xlabel("Trial Number")
    plt.ylabel("Cumulative Best WER")
    plt.title(f"WER Progress for {prec.__name__ if prec != nn.Linear else 'FullPrecision'}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"optuna_wer_progress_{prec.__name__ if prec != nn.Linear else 'FullPrecision'}.png")
    plt.close()

# Combine all results and create comparison plots
combined_df = pd.concat(all_results, ignore_index=True)

# Plot comparing WER across precision types
plt.figure(figsize=(10, 6))
for precision, group in combined_df.groupby("precision_type"):
    group = group.sort_values("trial_number")
    trial_nums = group["trial_number"].tolist()
    wer_values = group["trial_wer"].tolist()

    cum_best_wer = []
    current_best_wer = float("inf")
    for val in wer_values:
        if val is not None:
            current_best_wer = min(current_best_wer, val)
        cum_best_wer.append(current_best_wer)

    plt.plot(trial_nums, cum_best_wer, marker="o", label=precision)

plt.xlabel("Trial Number")
plt.ylabel("Cumulative Best WER")
plt.title("WER Progress by Precision Type")
plt.legend(title="Precision Type")
plt.grid(True)
plt.savefig("optuna_combined_wer_progress.png")
plt.close()

# Save the combined results to CSV
combined_df.to_csv("optuna_combined_results.csv", index=False)
print("Combined results saved to optuna_combined_results.csv")

# Run a standard evaluation on the best model for each precision
print("\nFinal Evaluation of Best Models:")
for prec in precision_choices:
    try:
        with open(f"best_model_{prec.__name__}.pkl", "rb") as f:
            best_model = dill.load(f)

        trainer = get_trainer(
            model=best_model,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            evaluate_metric="wer",
            num_train_epochs=0,  # No training, just evaluation
            data_collator=data_collator,
        )

        eval_results = trainer.evaluate()
        print(f"Best {prec.__name__} model:")
        print(f"WER: {eval_results['eval_wer']}")

        # Compute average bits
        best_model = best_model.cpu()
        graph = build_graph_from_model(best_model)
        graph, avg_bit_dict = calculate_avg_bits_mg_analysis_pass(graph, pass_args={})
        print(f"Average bits: {avg_bit_dict}")
        print("-" * 50)
    except Exception as e:
        print(f"Error evaluating best model for {prec.__name__}: {e}")