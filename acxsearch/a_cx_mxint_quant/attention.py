from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .attention_head import _ViTSelfAttentionHeadBase

from chop.nn.quantized.modules.linear import (
    LinearInteger,
)
from chop.nn.quantized.functional import fixed_softermax
from chop.nn.quantizers import integer_quantizer
from chop.nn.quantized.functional import matmul_integer

from typing import Optional, Tuple, Union

from .linear import MXIntLinear
from .attention_head import MXIntViTAttentionHead, IntViTAttentionHead
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

logger = get_logger(__name__)

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
        self.self_attention = MXIntViTAttentionHead(
            dim=self.head_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            q_config=q_config
        )

class IntAttention(_ViTAttentionBase):
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
        self.self_attention = IntViTAttentionHead(
            dim=self.head_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            q_config=q_config
        )

class QuantAttention(_ViTAttentionBase):
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
        
        if q_config.get('quant_type') == 'mxint':
            logger.debug("Using MXIntViTAttentionHead")
            self.self_attention = MXIntViTAttentionHead(
                dim=self.head_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                q_config=q_config
            )
        elif q_config.get('quant_type') == 'int':
            logger.debug("Using IntViTAttentionHead")
            self.self_attention = IntViTAttentionHead(
                dim=self.head_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                q_config=q_config)