from transformers.models.opt import OPTConfig, OPTForCausalLM, OPTModel

from .configuration_patched_opt import OPTConfigPatched
from .modeling_patched_opt import OPTForCausalLMPatched, OPTModelPatched
from .utils_patched_opt import (
    get_dummy_inputs_for_OPTForCausalLM,
    get_dummy_inputs_for_OPTModel,
)

opt_patched_task_to_model_cls = {
    "tokenizer": None,
    "config": OPTConfigPatched,
    "model": OPTModelPatched,
    "lm": OPTForCausalLMPatched,
    "cls": OPTModelPatched,
}

opt_patched_model_cls_to_dummy_input_fn = {
    OPTModelPatched: get_dummy_inputs_for_OPTModel,
    OPTForCausalLMPatched: get_dummy_inputs_for_OPTForCausalLM,
}

opt_patched_model_cls_to_original_model_cls = {
    OPTConfigPatched: OPTConfig,
    OPTModelPatched: OPTModel,
    OPTForCausalLMPatched: OPTForCausalLM,
}
