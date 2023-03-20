import inspect
from typing import Dict

import torch

from ..models import nlp_models, patched_nlp_models, vision_models
from ..models.patched_nlp_models import patched_model_cls_to_get_dummy_input


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


def get_dummy_inputs(model_name: str, task: str, model, data_loader):
    default_forward_kwargs = _get_default_args(model.forward)
    dummy_inputs = {}
    if (
        model_name in patched_nlp_models
        and type(model) in patched_model_cls_to_get_dummy_input
    ):
        dummy_input_fn = patched_model_cls_to_get_dummy_input[type(model)]
        dummy_inputs = dummy_input_fn()
        default_forward_kwargs.update(dummy_inputs)
        dummy_inputs = default_forward_kwargs
    elif model_name in patched_nlp_models or model_name in nlp_models:
        batch = next(iter(data_loader.train_dataloader))
        if task in ["cls", "classification", "lm", "language_modeling"]:
            dummy_inputs = {
                "input_ids": batch["input_ids"][[0], ...],
                "attention_mask": batch["attention_mask"][[0], ...],
            }
            default_forward_kwargs.update(dummy_inputs)
            dummy_inputs = default_forward_kwargs
        else:
            # translation
            dummy_inputs = {
                "input_ids": batch["input_ids"][[0], ...],
                "attention_mask": batch["attention_mask"][[0], ...],
                "decoder_input_ids": batch["attention_mask"][[0], ...],
                "decoder_attention_mask": batch["decoder_attention_mask"][[0], ...],
            }
            default_forward_kwargs.update(dummy_inputs)
            dummy_inputs = default_forward_kwargs
    elif model_name in vision_models:
        batch, _ = next(iter(data_loader.train_dataloader))
        assert (
            len(default_forward_kwargs) == 1
        ), "This vision module has more than 1 input"
        dummy_inputs = {}
    else:
        raise RuntimeError(f"Unsupported model+task: {model_name}+{task}")
    return dummy_inputs