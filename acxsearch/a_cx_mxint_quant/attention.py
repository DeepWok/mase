from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .attention_head import _ViTSelfAttentionHeadBase, ViTSelfAttentionHeadInteger

from chop.nn.quantized.modules.linear import (
    LinearInteger,
)
from chop.nn.quantized.functional import fixed_softermax
from chop.nn.quantizers import integer_quantizer
from chop.nn.quantized.functional import matmul_integer

from typing import Optional, Tuple, Union

from .linear import MXIntLinear
from .attention_head import MXIntViTAttentionHead

class _ViTAttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.self_attention = _ViTSelfAttentionHeadBase(
            dim=self.head_dim, num_heads=num_heads, attn_drop=attn_drop
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        def _tensor_reshape(x):
            return x.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = (
            _tensor_reshape(self.query(x)),
            _tensor_reshape(self.key(x)),
            _tensor_reshape(self.value(x)),
        )
        x = self.self_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTAttentionInteger(_ViTAttentionBase):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        q_config: dict = None,
        floor=True,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop)
        self.q_config = q_config
        self.query = LinearInteger(
            dim,
            dim,
            bias=qkv_bias,
            config={
                "data_in_width": q_config["data_in_width"],
                "data_in_frac_width": q_config["data_in_frac_width"],
                "weight_width": q_config["qkv_weight_width"],
                "weight_frac_width": q_config["qkv_weight_frac_width"],
                "bias_width": q_config["qkv_bias_width"],
                "bias_frac_width": q_config["qkv_bias_frac_width"],
            },
            out_config={
                "data_out_width": q_config["qkv_width"],
                "data_out_frac_width": q_config["qkv_frac_width"],
            },
            floor=floor,
        )
        self.key = LinearInteger(
            dim,
            dim,
            bias=qkv_bias,
            config={
                "data_in_width": q_config["data_in_width"],
                "data_in_frac_width": q_config["data_in_frac_width"],
                "weight_width": q_config["qkv_weight_width"],
                "weight_frac_width": q_config["qkv_weight_frac_width"],
                "bias_width": q_config["qkv_bias_width"],
                "bias_frac_width": q_config["qkv_bias_frac_width"],
            },
            out_config={
                "data_out_width": q_config["qkv_width"],
                "data_out_frac_width": q_config["qkv_frac_width"],
            },
            floor=floor,
        )
        self.value = LinearInteger(
            dim,
            dim,
            bias=qkv_bias,
            config={
                "data_in_width": q_config["data_in_width"],
                "data_in_frac_width": q_config["data_in_frac_width"],
                "weight_width": q_config["qkv_weight_width"],
                "weight_frac_width": q_config["qkv_weight_frac_width"],
                "bias_width": q_config["qkv_bias_width"],
                "bias_frac_width": q_config["qkv_bias_frac_width"],
            },
            out_config={
                "data_out_width": q_config["qkv_width"],
                "data_out_frac_width": q_config["qkv_frac_width"],
            },
            floor=floor,
        )
        self.self_attention = ViTSelfAttentionHeadInteger(
            dim=self.head_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            q_config={
                "query_width": q_config["qkv_width"],
                "query_frac_width": q_config["qkv_frac_width"],
                "key_width": q_config["qkv_width"],
                "key_frac_width": q_config["qkv_frac_width"],
                "value_width": q_config["qkv_width"],
                "value_frac_width": q_config["qkv_frac_width"],
                "qkmm_out_width": q_config["qkmm_out_width"],
                "qkmm_out_frac_width": q_config["qkmm_out_frac_width"],
                "softmax_exp_width": q_config["softmax_exp_width"],
                "softmax_exp_frac_width": q_config["softmax_exp_frac_width"],
                "softmax_out_frac_width": q_config["softmax_out_frac_width"],
                "svmm_out_width": q_config["svmm_out_width"],
                "svmm_out_frac_width": q_config["svmm_out_frac_width"],
            },
            floor=floor,
        )
        self.proj = LinearInteger(
            dim,
            dim,
            config={
                "data_in_width": q_config["svmm_out_width"],
                "data_in_frac_width": q_config["svmm_out_frac_width"],
                "weight_width": q_config["proj_weight_width"],
                "weight_frac_width": q_config["proj_weight_frac_width"],
                "bias_width": q_config["proj_bias_width"],
                "bias_frac_width": q_config["proj_bias_frac_width"],
            },
            out_config={
                "data_out_width": q_config["data_out_width"],
                "data_out_frac_width": q_config["data_out_frac_width"],
            },
            floor=floor,
        )

class MXIntAttention(_ViTAttentionBase):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        q_config: dict = None,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop)
        self.q_config = q_config
        
        # Replace attention with MXIntViTAttentionHead
        # self.self_attention = MXIntViTAttentionHead(
        #     dim=self.head_dim,
        #     num_heads=num_heads,
        #     attn_drop=attn_drop,
        #     q_config=q_config
        # )
