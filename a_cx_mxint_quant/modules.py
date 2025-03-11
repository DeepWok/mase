
import torch.nn as nn

from chop.nn.quantized.modules.attention import _ViTAttentionBase

import chop as chop
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

logger = get_logger(__name__)
set_logging_verbosity("debug")
from chop.models.vision.vit.vit import Attention
import torch
from mase_components.linear_layers.mxint_operators.test.utils import MXIntLinearHardware
class MXIntPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        q_config: dict = None,
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.q_config = q_config
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.norm = norm_layer(embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.distill_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), self.distill_token.expand(x.size(0), -1, -1), x), dim=1)
        return x

class ViTAttentionMxInt(_ViTAttentionBase):
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
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop)
        self.q_config = q_config


class MXIntAddition(nn.Module):
    def __init__(
        self,
        q_config,
    ) -> None:
        super().__init__()
        self.q_config = q_config
    
    def forward(self, x, y):
        return x + y

