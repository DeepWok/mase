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
from accelerate import init_empty_weights, init_on_device
from transformers import AutoTokenizer

from .configuration_llama import LlamaQuantizedConfig
from .modeling_llama import (
    LlamaQuantizedForCausalLM,
    LlamaQuantizedForSequenceClassification,
)


def get_llama_quant(
    name: str,
    task: str,
    info: dict,
    quant_config: dict | str,
    device: str = "meta",
    return_tokenizer: bool = False,
):
    # TODO: support cls tasks
    if task not in ["language_modeling", "lm"]:
        raise ValueError(f"Task {task} is not supported for Quantized Llama")

    match task:
        case "language_modeling" | "lm":
            with init_on_device(device):
                config = LlamaQuantizedConfig.from_pretrained(
                    name, quant_config=quant_config
                )
                model = LlamaQuantizedForCausalLM.from_pretrained(name, config=config)
        case _:
            raise ValueError(f"Task {task} is not supported for Quantized Llama")
    if not return_tokenizer:
        return model
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        return {"model": model, "tokenizer": tokenizer}
