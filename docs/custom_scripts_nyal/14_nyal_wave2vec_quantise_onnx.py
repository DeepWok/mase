import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

import logging
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

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
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
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
dataset_name = "nyalpatel/condensed_librispeech_asr"

# Logic inside get_tockenized_dataset needs to be improved using nyal's changes
tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    tokenizer_checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
    return_processor=True,
)

vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

model = AutoModelForCTC.from_pretrained(checkpoint)
# model.config.gradient_checkpointing = True
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head    # dynamic CTC head, separate this

# -------------------------------
# 2. Import dataset
# -------------------------------

batch_size = 2

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=checkpoint,
    num_workers=0,
    processor=processor
)
data_module.prepare_data()
data_module.setup()

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

smoothquant_config = {
    "smoothquant": True,
    "alpha": 0.5,
    "model": checkpoint,
    "task": "ctc",
    "dataset": dataset_name,
    "accelerator": "cuda",  
    "data_module": data_module,
    "batch_size": batch_size,
}


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
}


runtime_analysis_config = {
    "num_batches": 15,
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
    "ctc_head": ctc_head
}

mg_onnx, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)

mg_onnx, _ = passes.quantize_transform_pass(
    mg_onnx,
    pass_args=quantization_config,
)

_, results_onnx = runtime_analysis_pass(mg_onnx, pass_args=runtime_analysis_config)

print(f"Average WER", f"{results_onnx['Average WER']:.5g}")
print(f"Average Latency", f"{results_onnx['Average Latency']:.5g} ms")
print(f"Average RTF", f"{results_onnx['Average RTF']:.5g}")
print(f"Average GPU Power Usage", f"{results_onnx['Average GPU Power Usage']:.5g} W")
print(f"Inference Energy Consumption", f"{results_onnx['Inference Energy Consumption']:.5g} mWh")
