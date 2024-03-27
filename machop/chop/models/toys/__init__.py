import torch.nn as nn
from ..utils import MaseModelInfo
from .toy_custom_fn import get_toyfnnet
from .toy import (
    get_toy_tiny,
    get_toynet,
    get_toy_testmodel,
    get_toy_convnet,
    get_toy_emit,
)

TOY_MODELS = {
    "toy_tiny": {
        "model": get_toy_tiny,
        "info": MaseModelInfo(
            "toy_tiny",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
    "toy": {
        "model": get_toynet,
        "info": MaseModelInfo(
            "toy",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
    "toy_convnet": {
        "model": get_toy_convnet,
        "info": MaseModelInfo(
            "toy_convnet",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
    "toy_custom_fn": {
        "model": get_toyfnnet,
        "info": MaseModelInfo(
            "toy_custom_fn",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
    "toy_testmodel": {
        "model": get_toy_testmodel,
        "info": MaseModelInfo(
            "toy_testmodel",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
    "toy_emit": {
        "model": get_toy_emit,
        "info": MaseModelInfo(
            "toy_emit",
            model_source="toy",
            task_type="vision",
            image_classification=True,
            is_fx_traceable=True,
        ),
    },
}


def is_toy_model(name: str) -> bool:
    return name in TOY_MODELS


def get_toy_model_info(name: str) -> MaseModelInfo:
    if name not in TOY_MODELS:
        raise KeyError(f"Model {name} not found in toy models")
    return TOY_MODELS[name]["info"]


def get_toy_model(
    name: str, dataset_info: dict, pretrained: bool, **kwargs
) -> nn.Module:
    if name not in TOY_MODELS:
        raise KeyError(f"Model {name} not found in toy models")
    return TOY_MODELS[name]["model"](info=dataset_info, pretrained=pretrained, **kwargs)


def get_toy_model_cls(name: str) -> nn.Module:
    raise NotImplementedError
