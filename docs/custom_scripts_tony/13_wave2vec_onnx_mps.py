import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from pathlib import Path
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
from chop.dataset.nlp.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)

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

dataset_name = "nyalpatel/condensed_librispeech_asr"

data_module = MaseDataModule(
    name=dataset_name,
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
# 4. Initial Train & Evaluate
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

trainer.train()
eval_results = trainer.evaluate()

print("No ONNX Pass")
print(f"Evaluation WER: {eval_results['eval_wer']}")
print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Evaluation runtime: {eval_results['eval_runtime']}")
print(f"Evaluation samples per second: {eval_results['eval_samples_per_second']}")
print(f"Evaluation steps per second: {eval_results['eval_steps_per_second']}")

# -------------------------------
# 5. Add ONNX pass
# -------------------------------

smoothquant_config = {
    "smoothquant": True,               # Enable SmoothQuant in the ONNX pipeline
    "alpha": 0,                        # Smoothing parameter
    "model": checkpoint,               # Model identifier
    "task": "ctc",                     # Task name
    "dataset": dataset_name,           # Dataset name
    "accelerator": "cuda",             # Device for export
    "data_module": data_module,        # Data module for calibration
    "batch_size": 1,                   # Batch size for calibration
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

mg.model = ONNXWrapper(mg.model)

mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
print("ONNX Pass")
_, _ = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
print("ONNX Pass Numero 2")
_, _ = runtime_analysis_pass(onnx_meta['onnx_path'], pass_args=runtime_analysis_config)

