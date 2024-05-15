import torch.nn as nn
from ..utils import MaseModelInfo
from .nerf import NeRFModel

NERF_MODELS = {
    "nerf": {
        "model": NeRFModel,
        "info": MaseModelInfo(
            "nerf",
            model_source="nerf",
            task_type="nerf",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
}


def is_nerf_model(name: str) -> bool:
    return name in NERF_MODELS


def get_nerf_model_info(name: str) -> MaseModelInfo:
    if name not in NERF_MODELS:
        raise KeyError(f"Model {name} not found in NeRF models")
    return NERF_MODELS[name]["info"]


def get_nerf_model(name: str, dataset_info: dict, **kwargs) -> nn.Module:
    if name not in NERF_MODELS:
        raise KeyError(f"Model {name} not found in NeRF models")
    return NERF_MODELS[name]["model"](info=dataset_info)


def get_nerf_model_cls(name: str) -> nn.Module:
    raise NotImplementedError
