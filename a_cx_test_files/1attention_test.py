
from chop.nn.quantized.modules.attention_head import _ViTSelfAttentionHeadBase, ViTSelfAttentionHeadInteger
from chop.nn.quantized.modules.attention import _ViTAttentionBase

import torch.nn as nn
import torch

class ViTAttentionBase(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = torch.tensor(self.head_dim**-0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        attn = q @ k.transpose(-2, -1)
        attn = (attn * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == "__main__":
    dim = 4
    head = 2

    torch.manual_seed(0)
    x = torch.rand(1, dim, dim)
    module = ViTAttentionBase(dim, head)
    result = module(x)
    _module = _ViTAttentionBase(dim, head)
    _module.qkv.weight = module.qkv.weight
    _module.proj.weight = module.proj.weight
    _module.qkv.bias = module.qkv.bias
    _module.proj.bias = module.proj.bias
    _result = _module(x)
    print(result==_result)