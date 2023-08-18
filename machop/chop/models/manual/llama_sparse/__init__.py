from accelerate import init_empty_weights, init_on_device
from transformers import AutoTokenizer

from .configuration_llama_sparse import LlamaSparseConfig

from .modeling_llama_sparse import (
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


# %%
def get_llama_sparse(
    name: str,
    task: str,
    info: dict,
    # device: str = "meta",
    return_tokenizer: bool = False,
):
    # TODO: support cls tasks
    if task not in ["language_modeling", "lm"]:
        raise ValueError(f"Task {task} is not supported for Sparse Llama")

    match task:
        case "language_modeling" | "lm":
            # with init_on_device(device):
            config = LlamaSparseConfig.from_pretrained(name)
            model = LlamaForCausalLM.from_pretrained(name, config=config)

        case _:
            raise ValueError(f"Task {task} is not supported for Llama")
    if not return_tokenizer:
        return model
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        return {"model": model, "tokenizer": tokenizer}
