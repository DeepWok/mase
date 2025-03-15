import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
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
    
class MovementTrackingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.prev_params = {}
        self.movement_stats = {}
        model = kwargs["model"]
        for name, module in model.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                self.prev_params[name] = module.weight.detach().clone()
                self.movement_stats[name] = torch.zeros_like(module.weight)
                if not hasattr(module.weight, "metadata"):
                    module.metadata["weight"] = {}
                module.metadata["weight"]["stats"] = {"movement": self.movement_stats[name]}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, module in model.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                movement = (module.weight.detach() - self.prev_params[name]).abs()
                self.movement_stats[name] += movement
                self.prev_params[name].copy_(module.weight.detach())
                if not hasattr(module, "metadata"):
                    module.metadata = {}
                if "weight" not in module.metadata:
                    module.metadata["weight"] = {}
                module.metadata["weight"]["stats"] = {"movement": self.movement_stats[name]}

        return control

# -------------------------------
# 2. Define the MASE graph & movement metadata
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

for module in mg.model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
        if not hasattr(module, "metadata"):
            module.metadata = {}
        module.metadata["weight"] = {"stats": {"movement": torch.zeros_like(module.weight)}}

combined_model = CombinedWav2Vec2CTC(
    encoder=mg.model,
    ctc_head=ctc_head,
    decoder=decoder,
    beam_width=10
)

# -------------------------------
# 4. Warm-Up train the model
# -------------------------------

trainer = get_trainer(
    model=combined_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="wer",
    num_train_epochs=1,
    data_collator=data_collator,
    gradient_accumulation_steps = 4,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
)
trainer.add_callback(MovementTrackingCallback())

print("Starting warm-up training to accumulate movement data...")
# trainer.train()
print("Warm-up training complete.")

# -------------------------------
# 5. Prune the model
# -------------------------------

pruning_config = {
    "weight": {
        "sparsity": 0.0,
        "method": "l1-norm",
        "scope": "local",
        "granularity": "elementwise",
    },
    "activation": {
        "sparsity": 0.2,
        "method": "l1-norm",
        "scope": "local",
        "granularity": "elementwise",
    },
}

mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

print("Starting fine-tuning of the pruned model...")
trainer.train()

# Start evaluation
eval_results = trainer.evaluate()
print("Movement Pruning Pass")
print(f"Evaluation WER: {eval_results['eval_wer']}")
print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Evaluation runtime: {eval_results['eval_runtime']}")
print(f"Evaluation samples per second: {eval_results['eval_samples_per_second']}")
print(f"Evaluation steps per second: {eval_results['eval_steps_per_second']}")