from transformers.models.gpt2 import GPT2Tokenizer
from .configuration_opt import OPTQuantizedConfig
from .modeling_opt import (
    OPTQuantizedForCausalLM,
    OPTQuantizedForSequenceClassification,
    OPTQuantizedModel,
)
from .quant_config_opt import parse_opt_quantized_config
