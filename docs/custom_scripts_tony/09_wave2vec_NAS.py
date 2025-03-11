import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from chop.tools import get_tokenized_dataset # type: ignore
from chop.nn.modules import Identity
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.tools import get_trainer # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding
from pyctcdecode import build_ctcdecoder
import dill
import optuna
from optuna.samplers import GridSampler, TPESampler, RandomSampler
from chop.tools.utils import deepsetattr
import pandas as pd

# -------------------------------
# 1. Define the model and dataset
# -------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "librispeech_asr"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
tokenizer = processor.tokenizer

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

dataset = load_dataset(dataset_name, "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))

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

# -------------------------------
# 2 Construct Model
# -------------------------------

def construct_model(trial):
    config = AutoModelForCTC.from_pretrained(checkpoint).config

    config.num_hidden_layers = trial.suggest_categorical("num_layers", search_space["num_layers"])
    config.num_attention_heads = trial.suggest_categorical("num_heads", search_space["num_heads"])
    config.hidden_size = trial.suggest_categorical("hidden_size", search_space["hidden_size"])
    config.intermediate_size = trial.suggest_categorical("intermediate_size", search_space["intermediate_size"])

    model = AutoModelForCTC.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
    encoder = model.wav2vec2
    ctc_head = model.lm_head

    # print(model.config)

    # print(f"Encoder hidden size: {config.hidden_size}") # Debugging
    # print(f"CTC head input size: {model.lm_head.in_features}, output size: {model.lm_head.out_features}")

    # model.lm_head = nn.Linear(config.hidden_size, tokenizer.vocab_size)

    linear_choice = trial.suggest_categorical("linear_layer_choices", search_space["linear_layer_choices"])
    if linear_choice == "identity":
        for name, layer in encoder.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                deepsetattr(encoder, name, Identity())

    model = CombinedWav2Vec2CTC(encoder=encoder, ctc_head=ctc_head, decoder=decoder, beam_width=10)

    return model


def objective(trial):
    trial_model = construct_model(trial)

    trainer = get_trainer(
        model=trial_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        num_train_epochs=1,
        data_collator=data_collator,
    )

    # Train, evaluate and return
    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("model", trial_model)
    return eval_results['eval_wer']

# -------------------------------
# 2.1 Implement NAS
# -------------------------------

# Search space definition
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": ["linear", "identity"],
    "beam_width": [5, 10, 20],
}

# -------------------------------
# 3. Combine the models
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

        # if self.training:
        if labels is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            batch_size, time_steps, _ = logits.shape
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
            target_lengths = (labels != -100).sum(dim=1)

            loss = F.ctc_loss(log_probs, labels, input_lengths, target_lengths, blank=self.blank_id, reduction="mean", zero_infinity=True)
            output["loss"] = loss
        else:  # During evaluation/inference, decode instead of computing loss
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs.cpu().detach().numpy()

                # Decode each sample in batch
                pred_texts = [self.decoder.decode(lp, beam_width=self.beam_width).lower() for lp in log_probs_np]
                output["transcriptions"] = pred_texts

        return output

def run_study_and_get_curve(sampler, n_trials=None, study_name="study"):
    """
    Runs an Optuna study with the provided sampler and returns:
      - the study object
      - a list of best accuracies up to each trial (running max)
    """
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=60 * 60 * 24,
        show_progress_bar=True,
    )

    # Retains the minimum WER reached
    running_min_wer = []
    current_min = 1.0
    for t in study.trials:
        if t.value is not None and t.value < current_min:
            current_min = t.value
        running_min_wer.append(current_min)

    return study, running_min_wer

def save_study_results_to_csv(study, filename):
    """
    Saves each trial's results into a CSV, including:
      - trial number
      - objective value (accuracy)
      - parameters
      - model config parameters
    """
    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "wer": t.value,
        }
        # Merge in parameter key-value pairs directly
        row.update(t.params)

        # Add model config if it exists in user attributes
        if "model" in t.user_attrs:
            model_config = t.user_attrs["model"].config.to_dict()
            for key, value in model_config.items():
                row[f"config_{key}"] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved {len(rows)} trials with model configs to {filename}.")

if __name__ == "__main__":
    tpe_sampler = TPESampler()

    tpe_study, tpe_max_curve = run_study_and_get_curve(
       sampler=tpe_sampler,
       n_trials=10,
       study_name="wave2vec-tpe-study",
    )

    best_tpe_model = tpe_study.best_trial.user_attrs["model"].cpu()
    torch.save(best_tpe_model.state_dict(), "best_tpe_model.pth")
    print("✅ Best model saved successfully!")

    save_study_results_to_csv(tpe_study, "tpe_study_trials.csv")