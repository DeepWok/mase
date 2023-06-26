from transformers.models.opt import OPTConfig, OPTForCausalLM, OPTModel

from .configuration_opt_patched import OPTConfigPatched
from .modeling_opt_patched import OPTForCausalLMPatched, OPTModelPatched

# required input args will not be provided in dummy inputs before mase_symbolic_trace
opt_patched_model_cls_to_required_input_args = {
    OPTModelPatched: ["input_ids", "attention_mask"],
    OPTForCausalLMPatched: ["input_ids", "attention_mask", "labels"],
    # OPTForCausalLMPatched: ["input_ids", "attention_mask"],
}

# a mapping used to load pretrained original (un-patched) model
opt_patched_cls_to_original_cls = {
    OPTConfigPatched: OPTConfig,
    OPTModelPatched: OPTModel,
    OPTForCausalLMPatched: OPTForCausalLM,
}

# see 'name_to_final_module_map' in machop.session.plt_wrapper.nlp_classification
opt_patched_model_name_to_output_hidden_states_name = {
    "facebook/opt-125m@patched": "last_hidden_state",
    "facebook/opt-350m@patched": "last_hidden_state",
    "facebook/opt-1.3b@patched": "last_hidden_state",
    "facebook/opt-2.7b@patched": "last_hidden_state",
    "facebook/opt-6.7b@patched": "last_hidden_state",
    "facebook/opt-13b@patched": "last_hidden_state",
    "facebook/opt-30b@patched": "last_hidden_state",
    "facebook/opt-66b@patched": "last_hidden_state",
}

# a mapping used in get_patched_nlp_model
_opt_patched_task_to_model_cls = {
    "tokenizer": None,
    "config": OPTConfigPatched,
    "base": OPTModelPatched,
    "lm": OPTForCausalLMPatched,
    "cls": OPTModelPatched,
}
# opt_patched model create fn
opt_patched_name_to_patched_model_mapping = {
    "facebook/opt-125m@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-350m@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-1.3b@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-2.7b@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-6.7b@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-13b@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-30b@patched": _opt_patched_task_to_model_cls,
    "facebook/opt-66b@patched": _opt_patched_task_to_model_cls,
}

# pooler: ? -> hidden_size
opt_patched_model_name_to_pooler_size = {
    # for facebook/opt, ? is OPTConfig.word_embed_proj_dim
    "facebook/opt-125m@patched": (768, 768),
    "facebook/opt-350m@patched": (512, 1024),
    "facebook/opt-1.3b@patched": (2048, 2048),
    "facebook/opt-2.7b@patched": (2560, 2560),
    "facebook/opt-6.7b@patched": (4096, 4096),
    "facebook/opt-13b@patched": (5120, 5120),
    "facebook/opt-30b@patched": (7168, 7168),
    "facebook/opt-66b@patched": (9126, 9126),
}

# classifier: config.hidden_size -> num_classes
opt_patched_model_name_to_hidden_size = {
    "facebook/opt-125m@patched": 768,
    "facebook/opt-350m@patched": 1024,
    "facebook/opt-1.3b@patched": 2048,
    "facebook/opt-2.7b@patched": 2560,
    "facebook/opt-6.7b@patched": 4096,
    "facebook/opt-13b@patched": 5120,
    "facebook/opt-30b@patched": 7168,
    "facebook/opt-66b@patched": 9126,
}

_task_to_original_cls = {"cls": OPTModel, "lm": OPTForCausalLM}

opt_patched_model_name_to_task_to_original_cls = {
    "facebook/opt-125m@patched": _task_to_original_cls,
    "facebook/opt-350m@patched": _task_to_original_cls,
    "facebook/opt-1.3b@patched": _task_to_original_cls,
    "facebook/opt-2.7b@patched": _task_to_original_cls,
    "facebook/opt-6.7b@patched": _task_to_original_cls,
    "facebook/opt-13b@patched": _task_to_original_cls,
    "facebook/opt-30b@patched": _task_to_original_cls,
    "facebook/opt-66b@patched": _task_to_original_cls,
}
