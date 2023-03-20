import inspect
from typing import Dict

import torch

from ..models import nlp_models, vision_models

# --------------------
# Create dummy inputs
# --------------------


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def _create_input_ids(max_token_len: int):
    return torch.arange(max_token_len).long().reshape(1, max_token_len)


def _create_attention_mask():
    return torch.ones((1, 1), dtype=torch.bool)


NLP_MODELS_NO_DUMMY_INPUT_REQUIRED = ()

NLP_INPUTS_TO_BE_CREATED = {
    "bert-base-uncased": {"input_ids": _create_input_ids},
    "roberta-base": {"input_ids": _create_input_ids},
    # t5-small not working
    "t5-small": {
        "input_ids": _create_input_ids,
        "attention_mask": _create_attention_mask,
    },
    # facebook/opt-350 not working
    "facebook/opt-350m": {
        "input_ids": _create_input_ids,
        "attention_mask": _create_attention_mask,
    },
    # EleutherAI/gpt-neo-125M not working
    "EleutherAI/gpt-neo-125M": {
        "input_ids": _create_input_ids,
    },
}

ALL_CREATE_FN_ARGS_NAMES = {
    _create_input_ids: ("max_token_len",),
    _create_attention_mask: (),
}


def _nlp_create_required_inputs(model_name, kw_dict: Dict):
    required_inputs = {}
    if model_name in NLP_INPUTS_TO_BE_CREATED:
        model_inputs_to_be_updated = NLP_INPUTS_TO_BE_CREATED[model_name]
        if model_inputs_to_be_updated is None:
            return required_inputs
        else:
            for input_name, create_fn in model_inputs_to_be_updated.items():
                create_fn_arg_names = ALL_CREATE_FN_ARGS_NAMES[create_fn]
                create_fn_kwargs = {}
                for create_fn_arg_name in create_fn_arg_names:
                    create_fn_kwargs[create_fn_arg_name] = kw_dict[create_fn_arg_name]
                required_inputs[input_name] = create_fn(**create_fn_kwargs)
            return required_inputs
    else:
        raise RuntimeError(f"Unsupported NLP model {model_name} for creating graph")


def get_dummy_inputs(model_name: str, model, **kwargs):
    default_forward_kwargs = _get_default_args(model.forward)
    dummy_inputs = {}
    if model_name in nlp_models:
        if model_name in NLP_MODELS_NO_DUMMY_INPUT_REQUIRED:
            dummy_inputs = {}
        else:
            created_required_inputs = _nlp_create_required_inputs(model_name, kwargs)
            default_forward_kwargs.update(created_required_inputs)
            dummy_inputs = default_forward_kwargs
    elif model_name in vision_models:
        assert len(default_forward_kwargs) == 1, "This cnn module has more than 1 input"
        dummy_inputs = {}
    else:
        raise RuntimeError(f"Unsupported model name {model_name}")
    return dummy_inputs
