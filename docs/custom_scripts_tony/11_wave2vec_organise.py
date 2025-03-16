import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes import add_movement_metadata_analysis_pass
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.passes.graph.transforms.pruning import MovementTrackingCallback
from chop.tools import get_trainer # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
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

mg, _ = passes.add_movement_metadata_analysis_pass(mg)

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
    decoder=decoder,
    beam_width=10,
)
trainer.add_callback(MovementTrackingCallback())

print("Starting warm-up training to accumulate movement data...")
trainer.train()
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