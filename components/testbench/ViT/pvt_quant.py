from logging import getLogger
import sys

sys.path.append("/workspace/components/testbench/")
sys.path.append("/workspace/components/testbench/ViT")
from z_qlayers import quantize_to_int as q2i


import torch
import torch.nn as nn
import toml
from timm.models.layers import to_2tuple, trunc_normal_
import torch.nn.functional as F

__all__ = ["get_pvt_quant"]

from chop.models.manual.quant_utils import get_quantized_cls, get_quantized_func
from chop.passes.transforms.quantize.quantizers.integer import _integer_quantize
from ha_softmax import QHashSoftmax

logger = getLogger(__name__)


class fixed_affine(nn.Module):
    def __init__(self, config):
        super(fixed_affine, self).__init__()
        self.weight = torch.randn(1)
        self.bias = torch.randn(1)
        self.mult = get_quantized_func("mul", config["mul"])
        self.add = get_quantized_func("add", config["add"])
        self.config = config

    def forward(self, x):
        x = self.mult(x, self.weight, config=self.config["mul"])
        x = self.add(x, self.bias, config=self.config["add"])
        return x


class QuantizedAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        config=None,
        # sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.config = config
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = get_quantized_cls("linear", config["q_proj"])(
            dim, dim, bias=qkv_bias, config=config["q_proj"]
        )
        self.kv = get_quantized_cls("linear", config["q_proj"])(
            dim, dim * 2, bias=qkv_bias, config=config["kv_proj"]
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = get_quantized_cls("linear", config["z_proj"])(
            dim, dim, bias=True, config=config["z_proj"]
        )
        self.proj_drop = nn.Dropout(proj_drop)

        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    # def forward(self, x, H, W):
    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # if self.sr_ratio > 1:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     kv = (
        #         self.kv(x_)
        #         .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        #     )
        # else:
        kv = (
            self.kv(x)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = get_quantized_func("matmul", self.config["attn_matmul"])(
            q, k.transpose(-2, -1), self.config["attn_matmul"]
        )
        attn = QHashSoftmax(self.config["softmax"])(attn, self.scale)
        attn = self.attn_drop(attn)
        x = get_quantized_func("matmul", self.config["z_matmul"])(
            attn, v, self.config["z_matmul"]
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantizedMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        config=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = get_quantized_cls("linear", config["fc1_proj"])(
            in_features, hidden_features, bias=True, config=config["fc1_proj"]
        )
        self.act = get_quantized_func("relu", config["mlp_relu"])
        self.fc2 = get_quantized_cls("linear", config["fc2_proj"])(
            hidden_features, out_features, bias=True, config=config["fc2_proj"]
        )
        self.drop = nn.Dropout(drop)
        self.config = config

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x, config=self.config["mlp_relu"])
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# TODO: here
class QuantizedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        config,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        # sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = fixed_affine(config["affine_att"])
        self.attn = QuantizedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            # sr_ratio=sr_ratio,
            config=config["msa"],
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = fixed_affine(config["affine_mlp"])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = QuantizedMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            config=config["mlp"],
        )
        self.add1 = get_quantized_func("add", config["add1"])
        self.add2 = get_quantized_func("add", config["add2"])

        self.config = config

    def forward(self, x):
        x = self.add1(x, self.attn(self.norm1(x)), self.config["add1"])
        x = self.add2(x, self.mlp(self.norm2(x)), self.config["add2"])
        return x


class QuantizedPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, config=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = get_quantized_cls("conv2d", config["patch_proj"])(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            config=config["patch_proj"],
        )

        # TODO: layer NORM
        # self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class QuantizedPyramidVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        # norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        # pretrained_cfg=None,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.num_stages = num_stages
        for i in range(num_stages):
            patch_embed = QuantizedPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                config=config["patch_embed"],
            )
            num_patches = (
                patch_embed.num_patches
                if i != num_stages - 1
                else patch_embed.num_patches + 1
            )
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList(
                [
                    QuantizedBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        config=config["block"],
                        # sr_ratio=sr_ratios[i],
                    )
                    for j in range(depths[i])
                ]
            )
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
        # self.norm = fixed_affine(config["pvt_norm"])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        # classification head
        self.head = get_quantized_cls("linear", config["head_proj"])(
            in_features=embed_dims[3],
            out_features=num_classes,
            bias=True,
            config=config["head_proj"],
        )

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(
                        0, 3, 1, 2
                    ),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
            )

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                cls_tokens = _integer_quantize(
                    cls_tokens,
                    self.config["pos_add"]["data_in_width"],
                    self.config["pos_add"]["data_in_frac_width"],
                )
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            pos_add = get_quantized_func("add", self.config["pos_add"])(
                x, pos_embed, self.config["pos_add"]
            )
            x = pos_drop(pos_add)
            for blk in block:
                x = blk(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def get_pvt_quant(info, config, pretrained=False):
    num_classes = info["num_classes"]
    img_size = info["image_size"][2]
    in_chans = info["image_size"][0]
    config = toml.load(config)
    model = QuantizedPyramidVisionTransformer(
        num_classes=num_classes,
        img_size=img_size,
        in_chans=in_chans,
        config=config,
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        # num_heads=[1, 2, 5, 8],
        # mlp_ratios=[8, 8, 4, 4],
        # qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # depths=[2, 2, 2, 2],
        # sr_ratios=[8, 4, 2, 1],
    )
    # TODO: pretrained
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth",
    #         map_location="cpu",
    #         check_hash=True,
    #     )
    #     if num_classes != 1000:
    #         _ = checkpoint.pop("head.weight")
    #         _ = checkpoint.pop("head.bias")
    #         logger.warning(
    #             f"num_classes (={num_classes}) != 1000. The last classifier layer (head) is randomly initialized"
    #         )
    #     model.load_state_dict(checkpoint, strict=False)
    #     logger.info("Pretrained weights loaded into pvt_tiny")
    # else:
    #     logger.info("pvt_tiny randomly initialized")

    return model
