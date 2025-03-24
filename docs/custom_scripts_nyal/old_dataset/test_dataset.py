import torch
import torch.nn as nn
import torch.nn.functional as F
from chop.tools import get_tokenized_dataset  # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes  # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass  # type: ignore
from chop.tools import get_trainer  # type: ignore
from datasets import DatasetDict, Dataset, load_dataset
from chop.models import DataCollatorCTCWithPadding
from pyctcdecode import build_ctcdecoder
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Define the model and dataset
# -----------------------------------------------------------------------------
checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

processor = Wav2Vec2Processor.from_pretrained(checkpoint)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
tokenizer = processor.tokenizer

# Build vocabulary and decoder for CTC decoding
vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

# Use the CondensedLibrispeechASRDataset class from the new audio module
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset

dataset_path = Path("./preprocessed_data")
# NOTE: The constructor needs both path and dataset_path
# path is the base path for the dataset, dataset_path is used for the processed files
condensed_dataset = CondensedLibrispeechASRDataset(
    path=dataset_path,  # Provide the required path parameter
    dataset_path=dataset_path,  # Keep the existing dataset_path parameter
    split="train"
)
condensed_dataset.prepare_data()
condensed_dataset.setup()

# Create a data module instance for calibration and ONNX conversion
from chop.dataset import MaseDataModule

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=1,
    model_name=checkpoint,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# Define a preprocessing function for sample data
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

# Load and preprocess a small sample dataset using a valid split (e.g., "validation.clean")
sample_list = list(load_dataset(dataset_name, split="validation.clean").take(50))
small_dataset = Dataset.from_list(sample_list)
small_dataset = small_dataset.map(
    preprocess_function,
    remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
)

tokenized_dataset = DatasetDict({
    "train": small_dataset,
    "test": small_dataset
})

model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2   # static, FX-friendly
ctc_head = model.lm_head    # dynamic CTC head

# -----------------------------------------------------------------------------
# 2. Define the MASE graph and run metadata passes
# -----------------------------------------------------------------------------
mg = MaseGraph(
    encoder,
    hf_input_names=[
        "input_values",
        "attention_mask",
    ],
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

# -----------------------------------------------------------------------------
# 3. ONNX Export with SmoothQuant Enabled
# -----------------------------------------------------------------------------
# Define ONNX configuration with SmoothQuant enabled.
smoothquant_config = {
    "smoothquant": True,               # Enable SmoothQuant in the ONNX pipeline
    "alpha": 0.75,                     # Smoothing parameter
    "model": checkpoint,               # Model identifier
    "task": "ctc",                     # Task name
    "dataset": dataset_name,           # Dataset name
    "accelerator": "cuda",             # Device for export
    "data_module": data_module,        # Data module for calibration
    "batch_size": 1,                   # Batch size for calibration
}

# Import the ONNX Runtime Interface Pass from the interface submodule
from chop.passes.interface.onnxrt import onnx_runtime_interface_pass

# Run the ONNX export pass; this converts the PyTorch model (mg.model) into an ONNX model
# and applies SmoothQuant optimizations during pre-processing.
mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)

# onnx_meta is a dictionary containing keys like 'onnx_path' (the path to the ONNX model)

# -----------------------------------------------------------------------------
# 4. Further Quantization (Optional)
# -----------------------------------------------------------------------------
# If additional PTQ is desired, run the standard quantization pass.
quantization_config = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 8,
            "weight_width": 16,
            "weight_frac_width": 8,
            "bias_width": 16,
            "bias_frac_width": 8,
        }
    },
}

mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)

# -----------------------------------------------------------------------------
# 5. Combine the Optimized Encoder with the CTC Head and Train/Evaluate
# -----------------------------------------------------------------------------
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
            loss = F.ctc_loss(log_probs, labels, input_lengths, target_lengths, blank=self.blank_id, reduction="mean", zero_infinity=True)
            output["loss"] = loss
        else:
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                output["transcription"] = transcription
        return output

combined_model = CombinedWav2Vec2CTC(encoder=mg.model, ctc_head=ctc_head, decoder=decoder, beam_width=10)

trainer = get_trainer(
    model=combined_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="wer",
    num_train_epochs=1,
    data_collator=data_collator,
)
trainer.train()

eval_results = trainer.evaluate()

print("Quantising Pass")
print(f"Evaluation WER: {eval_results['eval_wer']}")
print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Evaluation runtime: {eval_results['eval_runtime']}")
print(f"Evaluation samples per second: {eval_results['eval_samples_per_second']}")
print(f"Evaluation steps per second: {eval_results['eval_steps_per_second']}")