import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from datasets import DatasetDict, load_dataset, Dataset
from pyctcdecode import build_ctcdecoder
import numpy as np

# Import MASE graph and passes (ensure PYTHONPATH is set correctly)
from chop import MaseGraph
import chop.passes as passes  # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass  # type: ignore
from chop.tools import get_trainer  # type: ignore

# ------------------------------------------------------------------------------
# Helper function: convert tensor to list if needed.
# ------------------------------------------------------------------------------
def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x

# ------------------------------------------------------------------------------
# Revised DataCollatorCTCWithPadding: manually pad sequences
# ------------------------------------------------------------------------------
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # Convert fields to lists
        input_values_list = [to_list(f["input_values"]) for f in features]
        attention_mask_list = [to_list(f["attention_mask"]) for f in features]
        label_list = [to_list(f["labels"]) for f in features]

        # Compute max lengths
        max_input_length = max(len(seq) for seq in input_values_list)
        max_label_length = max(len(seq) for seq in label_list)

        # Manually pad sequences
        padded_input_values = [
            seq + [0.0] * (max_input_length - len(seq)) for seq in input_values_list
        ]
        padded_attention_mask = [
            seq + [0] * (max_input_length - len(seq)) for seq in attention_mask_list
        ]
        padded_labels = [
            seq + [-100] * (max_label_length - len(seq)) for seq in label_list
        ]

        batch = {
            "input_values": torch.tensor(padded_input_values, dtype=torch.float),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch

# ------------------------------------------------------------------------------
# 1. Define the model and dataset
# ------------------------------------------------------------------------------
checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "librispeech_asr"

processor = Wav2Vec2Processor.from_pretrained(tokenizer_checkpoint)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
tokenizer = processor.tokenizer

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

# Load a small sample for quick experimentation
dataset = load_dataset(dataset_name, "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))

def preprocess_function(example):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    inputs = processor(audio=audio_array, sampling_rate=int(sampling_rate),
                       return_tensors="pt", padding=True)
    attention_mask = torch.ones(inputs.input_values.shape, dtype=torch.long)
    with processor.as_target_processor():
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids
    return {
        "input_values": inputs.input_values.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0)
    }

small_dataset = Dataset.from_list(sample_list)
small_dataset = small_dataset.map(preprocess_function,
                                  remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"])
tokenized_dataset = DatasetDict({
    "train": small_dataset,
    "test": small_dataset
})

model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head     # dynamic CTC head

# ------------------------------------------------------------------------------
# 2. Build the MASE graph for the encoder and initialize metadata
# ------------------------------------------------------------------------------
mg = MaseGraph(encoder, hf_input_names=["input_values", "attention_mask"])
mg, _ = passes.init_metadata_analysis_pass(mg)

dummy_in = {
    "input_values": torch.zeros((1, 16000), dtype=torch.float32),
    "attention_mask": torch.ones((1, 16000), dtype=torch.long),
}
mg, _ = passes.add_common_metadata_analysis_pass(mg,
                                                 pass_args={
                                                     "dummy_in": dummy_in,
                                                     "add_value": True,
                                                     "force_device_meta": False,
                                                 })

if not hasattr(mg.model, "metadata"):
    mg.model.metadata = {}
for name, param in mg.model.named_parameters():
    mg.model.metadata[name] = {"stats": {"movement": torch.zeros_like(param)}}

# ------------------------------------------------------------------------------
# 3. Build the combined model (without pruning applied yet)
# ------------------------------------------------------------------------------
class CombinedWav2Vec2CTC(nn.Module):
    def __init__(self, encoder, ctc_head, blank_id=0, beam_width=10, decoder=None):
        super().__init__()
        self.encoder = encoder 
        self.ctc_head = ctc_head  
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.decoder = decoder

    def forward(self, input_values, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = encoder_out["last_hidden_state"]
        logits = self.ctc_head(hidden_states)
        output = {"logits": logits, "labels": labels}
        if labels is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            batch_size, time_steps, _ = logits.shape
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
            target_lengths = (labels != -100).sum(dim=1)
            loss = F.ctc_loss(
                log_probs, labels, input_lengths, target_lengths,
                blank=self.blank_id, reduction="mean", zero_infinity=True
            )
            output["loss"] = loss
        else:
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                output["transcription"] = transcription  
        return output

combined_model = CombinedWav2Vec2CTC(encoder=mg.model, ctc_head=ctc_head, decoder=decoder, beam_width=10)

# ------------------------------------------------------------------------------
# 4. Define a Movement Tracking Callback to accumulate real movement data during training
# ------------------------------------------------------------------------------
class MovementTrackingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.prev_params = {}
        self.movement_stats = {}
        model = kwargs["model"]
        for name, param in model.encoder.named_parameters():
            self.prev_params[name] = param.detach().clone()
            self.movement_stats[name] = torch.zeros_like(param)
            if not hasattr(model.encoder, "metadata"):
                model.encoder.metadata = {}
            if name not in model.encoder.metadata:
                model.encoder.metadata[name] = {"stats": {}}
            model.encoder.metadata[name]["stats"]["movement"] = self.movement_stats[name]
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, param in model.encoder.named_parameters():
            movement = (param.detach() - self.prev_params[name]).abs()
            self.movement_stats[name] += movement
            self.prev_params[name].copy_(param.detach())
            model.encoder.metadata[name]["stats"]["movement"] = self.movement_stats[name]
        return control

# ------------------------------------------------------------------------------
# 5. Setup Trainer and run warm-up training to accumulate movement data
# ------------------------------------------------------------------------------
trainer = get_trainer(
    model=combined_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="wer",
    num_train_epochs=1,  # Warm-up training epoch
    data_collator=data_collator,
)
trainer.add_callback(MovementTrackingCallback())

print("Starting warm-up training to accumulate movement data...")
trainer.train()
print("Warm-up training complete.")

# ------------------------------------------------------------------------------
# 6. Apply movement-based pruning pass using the accumulated movement data
# ------------------------------------------------------------------------------
pruning_config = {
    "weight": {
        "sparsity": 0.2,
        "method": "movement",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.2,
        "method": "movement",
        "scope": "local",
    },
}

mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
_, _ = report_trainable_parameters_analysis_pass(mg.model)
combined_model.encoder = mg.model

# ------------------------------------------------------------------------------
# 7. Optionally, fine-tune the pruned model and evaluate
# ------------------------------------------------------------------------------
print("Starting fine-tuning of the pruned model...")
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation WER: {eval_results['eval_wer']}")
print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Evaluation runtime: {eval_results['eval_runtime']}")
print(f"Evaluation samples per second: {eval_results['eval_samples_per_second']}")
print(f"Evaluation steps per second: {eval_results['eval_steps_per_second']}")

