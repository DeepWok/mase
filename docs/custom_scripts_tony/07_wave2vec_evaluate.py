
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.tools import get_trainer # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding
from pyctcdecode import build_ctcdecoder

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

model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head     # dynamic CTC head, separate this

# -------------------------------
# 2. Define the MASE graph
# -------------------------------

mg = MaseGraph(
    encoder,
    hf_input_names=[
        "input_values",     # Or "input_ids",
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
                print("Logits shape:", logits.shape)
                print("Labels shape:", labels.shape)
                
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                output["transcription"] = transcription  
        return output

combined_model = CombinedWav2Vec2CTC(encoder=mg.model, ctc_head=ctc_head, decoder=decoder, beam_width=10)

# -------------------------------
# 4. Train the model
# -------------------------------

trainer = get_trainer(
    model=combined_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="wer",
    num_train_epochs=1,
    data_collator=data_collator,
)
trainer.train()

# Evaluate accuracy
eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")