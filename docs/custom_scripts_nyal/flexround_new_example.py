import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
from pyctcdecode import build_ctcdecoder

# CHOP imports
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger, get_trainer
from chop.passes.module import report_trainable_parameters_analysis_pass
from chop.passes.graph.transforms.quantize.flexround import FlexRoundQuantizer, apply_flexround_transform
from chop.passes.graph.transforms.quantize.quantize import quantize_transform_pass

# Ensure FlexRound keys are in the quantized module map
from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.flexround_modules import LinearFlexRound, Conv2dFlexRound, Conv1dFlexRound
quantized_module_map["linear_flexround"] = LinearFlexRound
quantized_module_map["conv2d_flexround"] = Conv2dFlexRound
quantized_module_map["conv1d_flexround"] = Conv1dFlexRound

logger = get_logger(__name__)
logger.setLevel("INFO")


def count_nonzero_parameters(model):
    """Count the actual non-zero parameters in the model."""
    total_params = 0
    nonzero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'parametrizations' not in name:
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    return total_params, nonzero_params


def print_parameter_count(model, description):
    """Helper function to count and print parameters."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total, nonzero = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    print(f"\n===== {description} =====")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total weight parameters: {total:,}")
    print(f"Non-zero weight parameters: {nonzero:,}")
    print(f"Sparsity: {sparsity:.2%}")
    return total_params, nonzero, sparsity


# Define a preprocessing function for the nyal dataset.
def preprocess_function(example):
    # example["audio"] is expected to be a dict with keys "array" and "sampling_rate"
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    inputs = processor(audio=audio_array, sampling_rate=int(sampling_rate), return_tensors="pt", padding=True)
    # Create an attention mask of ones (same shape as input_values)
    attention_mask = torch.ones(inputs.input_values.shape, dtype=torch.long)
    # Process the transcript text to get labels
    with processor.as_target_processor():
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids
    return {
        "input_values": inputs.input_values.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0)
    }


# -------------------------------
# Main script
# -------------------------------
def main():
    print("\n===== FlexRound Quantization Example =====")
    
    # Set checkpoint and dataset name
    checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "nyalpatel/condensed_librispeech_asr"  # nyal dataset identifier

    # Load the processor and model
    global processor
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = AutoModelForCTC.from_pretrained(checkpoint)
    
    # Split the model: use only the encoder for quantization; keep CTC head unquantized.
    encoder = model.wav2vec2  # static FX-friendly encoder
    ctc_head = model.lm_head   # dynamic CTC head (left in full precision)
    
    # Build a small evaluation sample from the nyal dataset.
    # Here we use the "validation.clean" split.
    sample_list = list(load_dataset(dataset_name, split="validation.clean").take(50))
    small_dataset = Dataset.from_list(sample_list)
    small_dataset = small_dataset.map(
        preprocess_function,
        remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
    )
    tokenized_dataset = DatasetDict({"train": small_dataset, "test": small_dataset})
    
    # Build a vocabulary and decoder for CTC (for decoding during inference)
    vocab = processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # -------------------------------
    # 1. Build MASE graph from the encoder only.
    # -------------------------------
    mg = MaseGraph(encoder, hf_input_names=["input_values", "attention_mask"])
    mg, _ = passes.init_metadata_analysis_pass(mg)
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg, _ = passes.add_common_metadata_analysis_pass(mg, pass_args={
        "dummy_in": dummy_in,
        "add_value": True,
        "force_device_meta": False,
    })
    
    before_params, before_nonzero, _ = print_parameter_count(mg.model, "Encoder BEFORE QUANTIZATION")
    
    # -------------------------------
    # 2. Apply FlexRound quantization using quantize_transform_pass on the encoder.
    # -------------------------------
    flexround_config = {
        "by": "type",
        "default": {"config": {"name": None}},  # leave unlisted ops unchanged
        "linear": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
        "conv2d": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
        "conv1d": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
    }
    
    mg_quant, _ = quantize_transform_pass(mg, flexround_config)
    after_params, after_nonzero, _ = print_parameter_count(mg_quant.model, "Encoder AFTER QUANTIZATION")
    
    # -------------------------------
    # 3. Combine the quantized encoder with the original CTC head.
    # -------------------------------
    class CombinedWav2Vec2CTC(nn.Module):
        def __init__(self, encoder, ctc_head, blank_id=0, beam_width=10, decoder=None):
            """
            Combines a quantized encoder with an unquantized CTC head.
            Args:
                encoder: Quantized encoder (result from mg_quant.model)
                ctc_head: Original full‑precision CTC head
                blank_id: Token ID for the blank symbol (typically 0)
                beam_width: Beam width for decoding (if using a decoder)
                decoder: (Optional) Beam search decoder (e.g., from pyctcdecode)
            """
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
                    log_probs,
                    labels,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_id,
                    reduction="mean",
                    zero_infinity=True,
                )
                output["loss"] = loss
            else:
                if self.decoder is not None:
                    log_probs = logits.log_softmax(dim=-1)
                    log_probs_np = log_probs[0].cpu().detach().numpy()
                    transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                    output["transcription"] = transcription
            return output

    combined_model = CombinedWav2Vec2CTC(encoder=mg_quant.model, ctc_head=ctc_head, decoder=decoder, beam_width=10)
    print("\n===== Combined Model Summary =====")
    print(combined_model)
    
    # -------------------------------
    # 4. Set up trainer using the nyal (Condensed LibriSpeech ASR) dataset.
    # -------------------------------
    # Use the preprocessed small_dataset from the "validation.clean" split.
    tokenized_dataset = DatasetDict({"train": small_dataset, "test": small_dataset})
    
    # Create a data collator for CTC padding.
    from chop.models import DataCollatorCTCWithPadding
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Set up trainer with WER as the evaluation metric.
    trainer = get_trainer(
        model=combined_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=processor.tokenizer,
        evaluate_metric="wer",
        num_train_epochs=1,
        data_collator=data_collator,
    )
    
    print("\n===== Starting Training =====")
    trainer.train()
    
    print("\n===== Evaluating Combined Model =====")
    eval_results = trainer.evaluate()
    print("Quantisation Pass Evaluation:")
    print(f"Evaluation WER: {eval_results.get('eval_wer', 'N/A')}")
    print(f"Evaluation Loss: {eval_results.get('eval_loss', 'N/A')}")
    print(f"Evaluation Runtime: {eval_results.get('eval_runtime', 'N/A')}")
    print(f"Evaluation Samples/s: {eval_results.get('eval_samples_per_second', 'N/A')}")
    print(f"Evaluation Steps/s: {eval_results.get('eval_steps_per_second', 'N/A')}")
    
    return combined_model

if __name__ == "__main__":
    combined_model = main()
