
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
    n = 3
    x = torch.rand(1, n, dim)
    qkv = nn.Linear(dim, 3 * dim)
    q = nn.Linear(dim, dim)
    k = nn.Linear(dim, dim)
    v = nn.Linear(dim, dim)
    
    new_weight = qkv.weight.reshape(3, -1, dim)
    new_bias = qkv.bias.reshape(3, -1, dim)
    q.weight,k.weight,v.weight  = nn.Parameter(new_weight[0]),nn.Parameter(new_weight[1]),nn.Parameter(new_weight[2])
    q.bias,k.bias,v.bias  = nn.Parameter(new_bias[0]),nn.Parameter(new_bias[1]),nn.Parameter(new_bias[2])
    qkv_x = qkv(x)
    qkv_x = qkv_x.reshape(-1, 3, dim).permute(1,0,2)
    print(qkv_x[0] == q(x))
    print(qkv_x[1] == k(x))
    print(qkv_x[2] == v(x))
    