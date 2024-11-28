import torch
import torch.nn as nn
from functools import partial
from logging import getLogger
from timm.layers import (
    get_act_layer,
    get_norm_layer,
    LayerType,
)
import numpy as np
from torchvision.models import ViT_B_16_Weights
from torchvision.models import VisionTransformer
from .utils import load_weights_from_npz

logger = getLogger(__name__)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    List,
)


def get_vit_tiny_patch16_224(info, pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    num_classes = info.num_classes
    img_size = info.image_size[-1]
    model = VisionTransformer(
        image_size=img_size,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=192,
        mlp_dim=768,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError(
            "Pretrained weights not available for vit_tiny_patch16"
        )
    return model


def get_vit_small_patch16_224(info, pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    num_classes = info.num_classes
    img_size = info.image_size[-1]
    model = VisionTransformer(
        image_size=img_size,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=384,
        mlp_dim=1536,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError(
            "Pretrained weights not available for vit_small_patch16"
        )
    return model


def get_vit_base_patch16_224(info, pretrained=False, **kwargs):
    """ViT-Base (Vit-B/16)"""
    num_classes = info.num_classes
    img_size = info.image_size[-1]

    model = VisionTransformer(
        image_size=img_size,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )
    if pretrained:
        pretrained_weight_cls = ViT_B_16_Weights.IMAGENET1K_V1
        pretrained_weight = pretrained_weight_cls.get_state_dict(progress=True)
        model.load_state_dict(pretrained_weight, strict=True)
        logger.info("Pretrained weights loaded into vit_base_patch16")
    else:
        logger.info("vit_base_patch16 randomly initialized")

    return model
