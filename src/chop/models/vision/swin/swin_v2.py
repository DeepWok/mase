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
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
from torchvision.models.swin_transformer import (
    SwinTransformer,
    SwinTransformerBlockV2,
    PatchMergingV2,
)

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


def get_swin_tiny_v2_224(info, pretrained=False, **kwargs):
    """Swin-Tiny"""

    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        pretrained_weight_cls = Swin_V2_T_Weights.IMAGENET1K_V1
        pretrained_weight = pretrained_weight_cls.get_state_dict(progress=True)
        model.load_state_dict(pretrained_weight, strict=True)
        logger.info("Pretrained weights loaded into swin_tiny")
    else:
        logger.info("swin_tiny randomly initialized")

    return model


def get_swin_small_v2_224(info, pretrained=False, **kwargs):
    """Swin-Small"""
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        pretrained_weight_cls = Swin_V2_S_Weights.IMAGENET1K_V1
        pretrained_weight = pretrained_weight_cls.get_state_dict(progress=True)
        model.load_state_dict(pretrained_weight, strict=True)
        logger.info("Pretrained weights loaded into swin_small")
    else:
        logger.info("swin_small randomly initialized")

    return model


def get_swin_base_v2_224(info, pretrained=False, **kwargs):
    """Swin-Base (Vit-B/16)"""
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        pretrained_weight_cls = Swin_V2_B_Weights.IMAGENET1K_V1
        pretrained_weight = pretrained_weight_cls.get_state_dict(progress=True)
        model.load_state_dict(pretrained_weight, strict=True)
        logger.info("Pretrained weights loaded into swin_base")
    else:
        logger.info("swin_base randomly initialized")

    return model
