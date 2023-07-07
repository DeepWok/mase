"""
This script shows how to quantize a model on the meta level.
Since the quantization is done on device="meta", no GPU memory is needed.
"""
import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(os.path.join("..", "..", "..", "..", "machop"))


from accelerate import init_empty_weights
from chop.models.manual.llama_quantized import (
    LlamaQuantizedConfig,
    LlamaQuantizedForCausalLM,
)
from transformers import AutoTokenizer

quant_configs = [
    "./quant_config_minimal.toml",
    "./quant_config_by_type.toml",
    "./quant_config_by_name.toml",
    "./quant_config_na.toml",
]
config = LlamaQuantizedConfig.from_pretrained(
    "lmsys/vicuna-7b-v1.3", quant_config=quant_configs[1]
)

with init_empty_weights():
    vicuna = LlamaQuantizedForCausalLM._from_config(config)

print(vicuna)
