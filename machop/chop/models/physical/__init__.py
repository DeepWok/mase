import torch.nn as nn
from ..utils import MaseModelInfo
from .jet_substructure import get_jsc_toy, get_jsc_tiny, get_jsc_s, get_jsc_txl

PHYSICAL_MODELS = {
    "jsc-toy": {
        "model": get_jsc_toy,
        "info": MaseModelInfo(
            "jsc-toy",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
    "jsc-tiny": {
        "model": get_jsc_tiny,
        "info": MaseModelInfo(
            "jsc-tiny",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
    "jsc-s": {
        "model": get_jsc_s,
        "info": MaseModelInfo(
            "jsc-s",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
    "jsc-txl": {
        "model": get_jsc_txl,
        "info": MaseModelInfo(
            "jsc-s",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
}


def is_physical_model(name: str) -> bool:
    return name in PHYSICAL_MODELS


def get_physical_model_info(name: str) -> MaseModelInfo:
    if name not in PHYSICAL_MODELS:
        raise KeyError(f"Model {name} not found in physical models")
    return PHYSICAL_MODELS[name]["info"]


def get_physical_model(name: str, dataset_info: dict, **kwargs) -> nn.Module:
    if name not in PHYSICAL_MODELS:
        raise KeyError(f"Model {name} not found in physical models")
    return PHYSICAL_MODELS[name]["model"](info=dataset_info)


def get_physical_model_cls(name: str) -> nn.Module:
    raise NotImplementedError
