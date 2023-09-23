import logging
from os import PathLike
import torch
import torch.nn as nn
from ..utils import MaseModelInfo
from .cswin import (
    get_cswin_64_small,
    get_cswin_64_tiny,
    get_cswin_96_base,
    get_cswin_144_large,
)
from .deit import (
    get_deit_base_patch16_224,
    get_deit_small_patch16_224,
    get_deit_tiny_patch16_224,
)
from .efficientnet import (
    get_efficientnet_b0,
    get_efficientnet_b3,
    get_efficientnet_v2_l,
    get_efficientnet_v2_m,
    get_efficientnet_v2_s,
)
from .mobilenet_v2 import get_mobilenet_v2
from .mobilenet_v3 import get_mobilenetv3_large, get_mobilenetv3_small
from .pvt import (
    get_pvt_large,
    get_pvt_medium,
    get_pvt_small,
    get_pvt_tiny,
    get_pvt_v2_b0,
    get_pvt_v2_b1,
    get_pvt_v2_b2,
    get_pvt_v2_b3,
    get_pvt_v2_b4,
    get_pvt_v2_b5,
)
from .resnet import (
    get_resnet18,
    get_resnet34,
    get_resnet50,
    get_resnet101,
    get_wide_resnet50_2,
)
from .wideresnet import wideresnet28_cifar

from .repvgg import (
    get_repvgg_a0,
    get_repvgg_a1,
    get_repvgg_a2,
    get_repvgg_b0,
    get_repvgg_b1,
    get_repvgg_b1g2,
    get_repvgg_b1g4,
    get_repvgg_b2,
    get_repvgg_b2g2,
    get_repvgg_b2g4,
    get_repvgg_b3,
    get_repvgg_b3g2,
    get_repvgg_b3g4,
    get_repvgg_d2se,
)

from .lfc import get_lfc

logger = logging.getLogger(__name__)

# fmt: off
VISION_MODELS = {
    # resnet
    "resnet18": {
        "get_model_fn_image_classification": get_resnet18,
        "info": MaseModelInfo(
            "resnet18", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "resnet34": {
        "get_model_fn_image_classification": get_resnet34,
        "info": MaseModelInfo(
            "resnet34", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "resnet50": {
        "get_model_fn_image_classification": get_resnet50,
        "info": MaseModelInfo(
            "resnet50", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "resnet101": {
        "get_model_fn_image_classification": get_resnet101,
        "info": MaseModelInfo(
            "resnet101", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # wide resnet
    "wideresnet50_2": {
        "get_model_fn_image_classification": get_wide_resnet50_2,
        "info": MaseModelInfo(
            "wideresnet50_2", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "wideresnet28_cifar": {
        "get_model_fn_image_classification": wideresnet28_cifar,
        "info": MaseModelInfo(
            "wideresnet28_cifar", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # mobilenet v2
    "mobilenetv2": {
        "get_model_fn_image_classification": get_mobilenet_v2,
        "info": MaseModelInfo(
            "mobilenetv2", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # mobilenet v3
    "mobilenetv3_small": {
        "get_model_fn_image_classification": get_mobilenetv3_small,
        "info": MaseModelInfo(
            "mobilenetv3_small", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "mobilenetv3_large": {
        "get_model_fn_image_classification": get_mobilenetv3_large,
        "info": MaseModelInfo(
            "mobilenetv3_large", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # efficient net
    "efficientnet_b0": {
        "get_model_fn_image_classification": get_efficientnet_b0,
        "info": MaseModelInfo(
            "efficientnet_b0", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "efficientnet_b3": {
        "get_model_fn_image_classification": get_efficientnet_b3,
        "info": MaseModelInfo(
            "efficientnet_b3", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "efficientnet_v2_s": {
        "get_model_fn_image_classification": get_efficientnet_v2_s,
        "info": MaseModelInfo(
            "efficientnet_v2_s", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "efficientnet_v2_m": {
        "get_model_fn_image_classification": get_efficientnet_v2_m,
        "info": MaseModelInfo(
            "efficientnet_v2_m", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "efficientnet_v2_l": {
        "get_model_fn_image_classification": get_efficientnet_v2_l,
        "info": MaseModelInfo(
            "efficientnet_v2_l", model_source="torchvision", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "pvt_tiny": {
        "get_model_fn_image_classification": get_pvt_tiny,
        "info": MaseModelInfo(
            "pvt_tiny", model_source="vision_others", task_type="vision", image_classification=True
        ), # TODO: Check if traceable
    },
    "pvt_small": {
        "get_model_fn_image_classification": get_pvt_small,
        "info": MaseModelInfo(
            "pvt_small", model_source="vision_others", task_type="vision", image_classification=True
        ), # TODO: Check if traceable
    },
    "pvt_medium": {
        "get_model_fn_image_classification": get_pvt_medium,
        "info": MaseModelInfo(
            "pvt_medium", model_source="vision_others", task_type="vision", image_classification=True
        ), # TODO: Check if traceable
    },
    "pvt_large": {
        "get_model_fn_image_classification": get_pvt_large,
        "info": MaseModelInfo(
            "pvt_large", model_source="vision_others", task_type="vision", image_classification=True
        ), # TODO: Check if traceable
    },
    "pvt_v2_b0": {
        "get_model_fn_image_classification": get_pvt_v2_b0,
        "info": MaseModelInfo(
            "pvt_v2_b0", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "pvt_v2_b1": {
        "get_model_fn_image_classification": get_pvt_v2_b1,
        "info": MaseModelInfo(
            "pvt_v2_b1", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "pvt_v2_b2": {
        "get_model_fn_image_classification": get_pvt_v2_b2,
        "info": MaseModelInfo(
            "pvt_v2_b2", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "pvt_v2_b3": {
        "get_model_fn_image_classification": get_pvt_v2_b3,
        "info": MaseModelInfo(
            "pvt_v2_b3", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "pvt_v2_b4": {
        "get_model_fn_image_classification": get_pvt_v2_b4,
        "info": MaseModelInfo(
            "pvt_v2_b4", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "pvt_v2_b5": {
        "get_model_fn_image_classification": get_pvt_v2_b5,
        "info": MaseModelInfo(
            "pvt_v2_b5", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # CSWin
    "cswin_64_tiny": {
        "get_model_fn_image_classification": get_cswin_64_tiny,
        "info": MaseModelInfo(
            "cswin_64_tiny", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "cswin_64_small": {
        "get_model_fn_image_classification": get_cswin_64_small,
        "info": MaseModelInfo(
            "cswin_64_small", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "cswin_96_base": {
        "get_model_fn_image_classification": get_cswin_96_base,
        "info": MaseModelInfo(
            "cswin_96_base", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "cswin_144_large": {
        "get_model_fn_image_classification": get_cswin_144_large,
        "info": MaseModelInfo(
            "cswin_144_large", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    # DeiT
    "deit_tiny_patch16_224": {
        "get_model_fn_image_classification": get_deit_tiny_patch16_224,
        "info": MaseModelInfo(
            "deit_tiny_patch16_224", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "deit_small_patch16_224": {
        "get_model_fn_image_classification": get_deit_small_patch16_224,
        "info": MaseModelInfo(
            "deit_small_patch16_224", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    "deit_base_patch16_224": {
        "get_model_fn_image_classification": get_deit_base_patch16_224,
        "info": MaseModelInfo(
            "deit_base_patch16_224", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    },
    # repvgg
    # TODO: Check if traceable
    "repvgg_a0": {
        "get_model_fn_image_classification": get_repvgg_a0,
        "info": MaseModelInfo(
            "repvgg_a0", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_a1": {
        "get_model_fn_image_classification": get_repvgg_a1,
        "info": MaseModelInfo(
            "repvgg_a1", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_a2": {
        "get_model_fn_image_classification": get_repvgg_a2,
        "info": MaseModelInfo(
            "repvgg_a2", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b0": {
        "get_model_fn_image_classification": get_repvgg_b0,
        "info": MaseModelInfo(
            "repvgg_b0", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b1": {
        "get_model_fn_image_classification": get_repvgg_b1,
        "info": MaseModelInfo(
            "repvgg_b1", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b1g2": {
        "get_model_fn_image_classification": get_repvgg_b1g2,
        "info": MaseModelInfo(
            "repvgg_b1g2", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b1g4": {
        "get_model_fn_image_classification": get_repvgg_b1g4,
        "info": MaseModelInfo(
            "repvgg_b1g4", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b2": {
        "get_model_fn_image_classification": get_repvgg_b2,
        "info": MaseModelInfo(
            "repvgg_b2", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b2g2": {
        "get_model_fn_image_classification": get_repvgg_b2g2,
        "info": MaseModelInfo(
            "repvgg_b2g2", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b2g4": {
        "get_model_fn_image_classification": get_repvgg_b2g4,
        "info": MaseModelInfo(
            "repvgg_b2g4", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b3": {
        "get_model_fn_image_classification": get_repvgg_b3,
        "info": MaseModelInfo(
            "repvgg_b3", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b3g2": {
        "get_model_fn_image_classification": get_repvgg_b3g2,
        "info": MaseModelInfo(
            "repvgg_b3g2", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_b3g4": {
        "get_model_fn_image_classification": get_repvgg_b3g4,
        "info": MaseModelInfo(
            "repvgg_b3g4", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "repvgg_d2se": {
        "get_model_fn_image_classification": get_repvgg_d2se,
        "info": MaseModelInfo(
            "repvgg_d2se", model_source="vision_others", task_type="vision", image_classification=True
        ),
    },
    "lfc": {
        "get_model_fn_image_classification": get_lfc,
        "info": MaseModelInfo(
            "lfc", model_source="vision_others", task_type="vision", image_classification=True, is_fx_traceable=True
        ),
    }
}
# fmt: on


def is_vision_model(name: str) -> bool:
    return name in VISION_MODELS


def get_vision_model_info(name: str) -> MaseModelInfo:
    return VISION_MODELS[name]["info"]


def get_vision_model(
    name: str,
    task: str,
    dataset_info: dict,
    pretrained: bool,
    checkpoint: PathLike = None,
):
    """
    Args:
        name: The name of the model.
        task: The task type.
        info: The model info.
        pretrained: Whether to load the pretrained model dict.
        checkpoint: The path to the checkpoint.

    ---
    A vision model checkpoint includes only model dict.
    - if `pretrained` is False and `checkpoint` is None, we will initialize the model randomly.
    - if `pretrained` is True and `checkpoint` is None, we will load the model dict from torchvision or other sources.
    - if `pretrained` is True and `checkpoint` is not None, we will load the model dict from `checkpoint`. A warning will be raised if `pretrained`.
    - if `pretrained` is False and `checkpoint` is not None, we will load the model dict from `checkpoint`.
    """
    if pretrained and checkpoint is not None:
        logger.warning(
            "Both `pretrained` and `checkpoint` are specified. `pretrained` will be ignored."
        )
        pretrained = False
    if name not in VISION_MODELS:
        raise ValueError(f"Vision model {name} is not supported")

    model_info = VISION_MODELS[name]["info"]

    match task:
        case "cls" | "classification":
            assert (
                model_info.image_classification
            ), f"Task `{task}` is not supported for model `{name}`"
            model = VISION_MODELS[name]["get_model_fn_image_classification"](
                info=dataset_info, pretrained=pretrained
            )
            if checkpoint:
                model.load_state_dict(torch.load(checkpoint))
        case _:
            raise ValueError(f"Task `{task}` is not supported for model `{name}`")

    return model


def get_vision_model_cls(name: str):
    raise NotImplementedError
