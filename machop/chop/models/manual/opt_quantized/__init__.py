# from transformers.models.opt import OPTConfig, OPTForCausalLM, OPTModel
from transformers import AutoTokenizer
from .configuration_opt import OPTQuantizedConfig
from .modeling_opt import (
    OPTQuantizedForCausalLM,
    OPTQuantizedForSequenceClassification,
    OPTQuantizedModel,
)

# required input args will not be provided in dummy inputs before mase_symbolic_trace
name_to_input_args_map = {
    OPTQuantizedModel: ["input_ids", "attention_mask"],
    OPTQuantizedForCausalLM: ["input_ids", "attention_mask", "labels"],
}

# a mapping used to load pretrained original (un-patched) model

# a mapping used in get_patched_nlp_model
_opt_task_to_model_cls = {
    "tokenizer": None,
    "config": OPTQuantizedConfig,
    "base": OPTQuantizedModel,
    "lm": OPTQuantizedForCausalLM,
    "cls": OPTQuantizedModel,
}
# opt_patched model create fn
name_to_model_map = {
    "opt-125m-quantized": _opt_task_to_model_cls,
    "opt-350m-quantized": _opt_task_to_model_cls,
    "opt-1.3b-quantized": _opt_task_to_model_cls,
    "opt-2.7b-quantized": _opt_task_to_model_cls,
    "opt-6.7b-quantized": _opt_task_to_model_cls,
    "opt-13b-quantized": _opt_task_to_model_cls,
    "opt-30b-quantized": _opt_task_to_model_cls,
    "opt-66b-quantized": _opt_task_to_model_cls,
}

_task_to_original_cls = {"cls": OPTQuantizedModel, "lm": OPTQuantizedForCausalLM}

name_to_cls_map = {
    "opt-125m-quantized": _task_to_original_cls,
    "opt-350m-quantized": _task_to_original_cls,
    "opt-1.3b-quantized": _task_to_original_cls,
    "opt-2.7b-quantized": _task_to_original_cls,
    "opt-6.7b-quantized": _task_to_original_cls,
    "opt-13b-quantized": _task_to_original_cls,
    "opt-30b-quantized": _task_to_original_cls,
    "opt-66b-quantized": _task_to_original_cls,
}

name_hash = {
    "opt-125m-quantized": "facebook/opt-125m",
    "opt-350m-quantized": "facebook/opt-350m",
    "opt-1.3b-quantized": "facebook/opt-1.3b",
    "opt-2.7b-quantized": "facebook/opt-2.7b",
    "opt-6.7b-quantized": "facebook/opt-6.7b",
    "opt-13b-quantized": "facebook/opt-13b",
    "opt-30b-quantized": "facebook/opt-30b",
    "opt-66b-quantized": "facebook/opt-66b",
}


def get_opt_quantized(
    name: str,
    task: str,
    info: dict,
    # device: str = "meta",
    return_tokenizer: bool = False,
):
    # TODO: support cls tasks
    if task not in ["language_modeling", "lm", "cls", "classification"]:
        raise ValueError(f"Task {task} is not supported for plain opt")

    match task:
        case "language_modeling" | "lm":
            # with init_on_device(device):
            config, model = (
                _opt_task_to_model_cls["config"],
                _opt_task_to_model_cls["lm"],
            )
        case "classification" | "cls":
            config, model = (
                _opt_task_to_model_cls["config"],
                _opt_task_to_model_cls["cls"],
            )
        case _:
            raise ValueError(f"Task {task} is not supported for Llama")
    hashed_name = name_hash[name]
    config = config.from_pretrained(hashed_name)
    model = model.from_pretrained(hashed_name, config=config)
    if not return_tokenizer:
        return model
    else:
        tokenizer = AutoTokenizer.from_pretrained(hashed_name)
        return {"model": model, "tokenizer": tokenizer}
