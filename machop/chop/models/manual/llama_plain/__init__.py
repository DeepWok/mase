from accelerate import init_empty_weights, init_on_device
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama import LlamaTokenizer

from .configuration_llama import LlamaConfig
from .modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

"""
Vicuna: https://huggingface.co/lmsys
- Vicuna
    - lmsys/vicuna-7b-v1.3
    - lmsys/vicuna-13b-v1.3
    - lmsys/vicuna-33b-v1.3
- Llama
    - huggyllama/llama-7b
    - huggyllama/llama-13b
    - huggyllama/llama-30b
    - huggyllama/llama-65b
"""
