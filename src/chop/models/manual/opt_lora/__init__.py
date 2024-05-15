# from transformers.models.opt import OPTConfig, OPTForCausalLM, OPTModel

from .configuration_opt_lora import OPTLoraConfig
from .modeling_opt_lora import OPTForCausalLM, OPTModel
from transformers import AutoTokenizer

# required input args will not be provided in dummy inputs before mase_symbolic_trace
name_to_input_args_map = {
    OPTModel: ["input_ids", "attention_mask"],
    OPTForCausalLM: ["input_ids", "attention_mask", "labels"],
}

# a mapping used to load pretrained original (un-patched) model

# see 'name_to_final_module_map' in machop.session.plt_wrapper.nlp_classification
name_to_hidden_module_map = {
    "facebook/opt-125m": "last_hidden_state",
    "facebook/opt-350m": "last_hidden_state",
    "facebook/opt-1.3b": "last_hidden_state",
    "facebook/opt-2.7b": "last_hidden_state",
    "facebook/opt-6.7b": "last_hidden_state",
    "facebook/opt-13b": "last_hidden_state",
    "facebook/opt-30b": "last_hidden_state",
    "facebook/opt-66b": "last_hidden_state",
}

# a mapping used in get_patched_nlp_model
_opt_task_to_model_cls = {
    "tokenizer": None,
    "config": OPTLoraConfig,
    "base": OPTModel,
    "lm": OPTForCausalLM,
    "cls": OPTModel,
}
# opt_patched model create fn
name_to_model_map = {
    "facebook/opt-125m": _opt_task_to_model_cls,
    "facebook/opt-350m": _opt_task_to_model_cls,
    "facebook/opt-1.3b": _opt_task_to_model_cls,
    "facebook/opt-2.7b": _opt_task_to_model_cls,
    "facebook/opt-6.7b": _opt_task_to_model_cls,
    "facebook/opt-13b": _opt_task_to_model_cls,
    "facebook/opt-30b": _opt_task_to_model_cls,
    "facebook/opt-66b": _opt_task_to_model_cls,
}

# pooler: ? -> hidden_size
name_to_pooler_size_map = {
    # for facebook/opt, ? is OPTConfig.word_embed_proj_dim
    "facebook/opt-125": (768, 768),
    "facebook/opt-350": (512, 1024),
    "facebook/opt-1.3": (2048, 2048),
    "facebook/opt-2.7": (2560, 2560),
    "facebook/opt-6.7": (4096, 4096),
    "facebook/opt-13b": (5120, 5120),
    "facebook/opt-30b": (7168, 7168),
    "facebook/opt-66b": (9126, 9126),
}

# classifier: config.hidden_size -> num_classes
name_to_hidden_size_map = {
    "facebook/opt-125m": 768,
    "facebook/opt-350m": 1024,
    "facebook/opt-1.3b": 2048,
    "facebook/opt-2.7b": 2560,
    "facebook/opt-6.7b": 4096,
    "facebook/opt-13b": 5120,
    "facebook/opt-30b": 7168,
    "facebook/opt-66b": 9126,
}

_task_to_original_cls = {"cls": OPTModel, "lm": OPTForCausalLM}

name_to_cls_map = {
    "facebook/opt-125m": _task_to_original_cls,
    "facebook/opt-350m": _task_to_original_cls,
    "facebook/opt-1.3b": _task_to_original_cls,
    "facebook/opt-2.7b": _task_to_original_cls,
    "facebook/opt-6.7b": _task_to_original_cls,
    "facebook/opt-13b": _task_to_original_cls,
    "facebook/opt-30b": _task_to_original_cls,
    "facebook/opt-66b": _task_to_original_cls,
}


def get_opt_plain(
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
    config = config.from_pretrained(name)
    model = model.from_pretrained(name, config=config)
    if not return_tokenizer:
        return model
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        return {"model": model, "tokenizer": tokenizer}
