import torch
import torch.nn as nn
import torch.fx as fx
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.tools import get_trainer # type: ignore
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.models import DataCollatorCTCWithPadding

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "librispeech_asr"

dataset, tokenizer = get_tokenized_dataset(      # Didn't have enough memory for this
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head     # dynamic CTC head, separate this

# print(model.config)
# model.config.problem_type = ""    Not needed for CTC

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

mg.draw()

_, _ = report_trainable_parameters_analysis_pass(mg.model)
