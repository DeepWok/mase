import inspect

from chop.models import (
    nlp_models,
    patched_model_cls_to_required_input_args,
    patched_nlp_models,
    vision_models,
)


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def get_cf_args(model_name: str, task: str, model):
    default_forward_kwargs = _get_default_args(model.forward)
    cf_args = {}
    if (
        model_name in patched_nlp_models
        and type(model) in patched_model_cls_to_required_input_args
    ):
        required_input_args = patched_model_cls_to_required_input_args[type(model)]
        for required_input_arg in required_input_args:
            default_forward_kwargs.pop(required_input_arg)
        cf_args = default_forward_kwargs
    elif model_name in patched_nlp_models or model_name in nlp_models:
        if task in ["cls", "classification"]:
            required_input_args = ["input_ids", "attention_mask"]
        elif task in ["lm", "language_modeling"]:
            required_input_args = ["input_ids", "attention_mask", "labels"]
            # required_input_args = ["input_ids", "attention_mask"]
        else:
            # translation
            required_input_args = [
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
        for required_input_arg in required_input_args:
            default_forward_kwargs.pop(required_input_arg)
        cf_args = default_forward_kwargs
    elif model_name in vision_models:
        # Currently the only input to vision model is a Tensor x
        cf_args = {}
    else:
        raise RuntimeError(f"Unsupported model+task: {model_name}+{task}")
    return cf_args
