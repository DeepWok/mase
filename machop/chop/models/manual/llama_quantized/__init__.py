"""
---
HuggingFace checkpoints
- Vicuna
    - lmsys/vicuna-7b-v1.3
    - lmsys/vicuna-13b-v1.3
    - lmsys/vicuna-33b-v1.3
- Llama
    - huggyllama/llama-7b
    - huggyllama/llama-13b
    - huggyllama/llama-30b
    - huggyllama/llama-65b
- Small llama models
    - Cheng98/llama-160m

---
Example usage:
```python
config = LlamaQuantizedConfig.from_pretrained(
    "lmsys/vicuna-7b-v1.3", quant_config="./quant_config_minimal.toml"
)
max_memory_mapping = {
    0: "23GB",
    1: "23GB",
    2: "23GB",
    3: "23GB",
} # model parallelism
vicuna = LlamaQuantizedForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.3",
    config=config,
    device_map="auto",
    max_memory=max_memory_mapping,
)
```
"""
from transformers.models.llama import LlamaTokenizer

from .configuration_llama import LlamaQuantizedConfig
from .modeling_llama import (
    LlamaQuantizedForCausalLM,
    LlamaQuantizedForSequenceClassification,
)

from .quant_config_llama import parse_llama_quantized_config
