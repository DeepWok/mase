# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from logging import getLogger

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer, _cfg

logger = getLogger(__name__)
# __all__ = [
#     "deit_tiny_patch16_224",
#     "deit_small_patch16_224",
#     "deit_base_patch16_224",
#     "deit_tiny_distilled_patch16_224",
#     "deit_small_distilled_patch16_224",
#     "deit_base_distilled_patch16_224",
#     "deit_base_patch16_384",
#     "deit_base_distilled_patch16_384",
# ]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


def get_deit_tiny_patch16_224(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = VisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_tiny_patch16_224")
    else:
        logger.info("deit_tiny_patch16_224 randomly initialized")
    return model


def get_deit_small_patch16_224(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = VisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_small_patch16_224")
    else:
        logger.info("deit_small_patch16_224 randomly initialized")
    return model


def get_deit_base_patch16_224(info, pretrained=False, **kwargs):
    kwargs.pop("info")
    num_classes = info.num_classes
    model = VisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_base_patch16_224")
    else:
        logger.info("deit_base_patch16_224 randomly initialized")
    return model


def get_deit_tiny_distilled_patch16_224(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = DistilledVisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_tiny_patch16_224")
    else:
        logger.info("deit_base_patch16_224 randomly initialized")
    return model


def get_deit_small_distilled_patch16_224(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = DistilledVisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_small_distilled_patch16_224")
    else:
        logger.info("deit_small_distilled_patch16_224 randomly initialized")
    return model


def get_deit_base_distilled_patch16_224(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = DistilledVisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_base_distilled_patch16_224")
    else:
        logger.info("deit_base_distilled_patch16_224 randomly initialized")
    return model


def get_deit_base_patch16_384(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = VisionTransformer(
        num_classes=num_classes,
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )

        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_base_patch16_384")
    else:
        logger.info("deit_base_patch16_384 randomly initialized")
    return model


def get_deit_base_distilled_patch16_384(info, pretrained=False, **kwargs):
    num_classes = info.num_classes
    model = DistilledVisionTransformer(
        num_classes=num_classes,
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu",
            check_hash=True,
        )
        checkpoint = checkpoint["model"]
        if num_classes != 1000:
            _ = checkpoint.pop("head.weight")
            _ = checkpoint.pop("head.bias")
            logger.warning(
                f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
            )
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded into deit_base_distilled_patch16_384")
    else:
        logger.info("deit_base_distilled_patch16_384 randomly initialized")
    return model
