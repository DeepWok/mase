import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from typing import Callable
import chop.nn.snn.base as base
import chop.nn.snn.functional as functional

from chop.nn.snn.modules import Conv2d, BatchNorm2d
import chop.nn.snn.modules.surrogate as surrogate
from chop.nn.snn.modules.neuron import LIFNode, ParametricLIFNode


class Conv1x1(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
            step_mode="m",
        )


class LIF(LIFNode):
    def __init__(self):
        super().__init__(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend="cupy",
            store_v_seq=False,
        )


class PLIF(ParametricLIFNode):
    def __init__(self):
        super().__init__(
            init_tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend="cupy",
            store_v_seq=False,
        )


class BN(BatchNorm2d):
    """
    BatchNorm2d with added extra warning message for input shape check.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        step_mode="m",
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, step_mode
        )

    def forward(self, x: Tensor):
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return super().forward(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == "l" or spike == "r" or spike == "both"
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            step_mode="m",
        )


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(),
                    Conv3x3(
                        inner_channels,
                        inner_channels,
                        groups=inner_channels // group_size,
                    ),
                    BN(inner_channels),
                )
            )
        self.down = nn.Sequential(
            activation(),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size, activation=LIF):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.register_buffer("firing_rate_x", torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer("firing_rate_attn", torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = activation()

        self.W = Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode="m")
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul("r")
        self.matmul2 = SpikingMatmul("r")
        self.activation_attn = activation()
        self.activation_out = activation()

        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = (
            y[:, :, :, : C // self.num_heads, :],
            y[:, :, :, C // self.num_heads :, :],
        )
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                1 - self.momentum
            )
        scale1 = 1.0 / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = (
                self.firing_rate_attn * self.momentum
                + firing_rate_attn * (1 - self.momentum)
            )
        scale2 = 1.0 / torch.sqrt(self.firing_rate_attn * self.lenth)
        out = self.matmul2(y2, attn)
        out = out * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)

        out = self.Wproj(out)
        out = self.norm_proj(out)
        out = out + x_feat
        return out
