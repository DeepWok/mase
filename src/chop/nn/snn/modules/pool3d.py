import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn.common_types import (
    _size_any_t,
    _size_1_t,
    _size_2_t,
    _size_3_t,
    _ratio_any_t,
)
from typing import Optional, List, Tuple, Union
from typing import Callable
import chop.nn.snn.base as base
import chop.nn.snn.functional as functional


class MaxPool3d(nn.MaxPool3d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        step_mode="s",
    ) -> None:
        """
        * :ref:`API in English <MaxPool3d-en>`

        .. _MaxPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool3d`

        * :ref:`中文 API <MaxPool3d-cn>`

        .. _MaxPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool3d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            x = super().forward(x)

        elif self.step_mode == "m":
            if x.dim() != 6:
                raise ValueError(
                    f"expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!"
                )
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AvgPool3d(nn.AvgPool3d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        step_mode="s",
    ) -> None:
        """
        * :ref:`API in English <AvgPool3d-en>`

        .. _AvgPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool3d`

        * :ref:`中文 API <AvgPool3d-cn>`

        .. _AvgPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool3d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            x = super().forward(x)

        elif self.step_mode == "m":
            if x.dim() != 6:
                raise ValueError(
                    f"expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!"
                )
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, base.StepModule):
    def __init__(self, output_size, step_mode="s") -> None:
        """
        * :ref:`API in English <AdaptiveAvgPool3d-en>`

        .. _AdaptiveAvgPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool3d`

        * :ref:`中文 API <AdaptiveAvgPool3d-cn>`

        .. _AdaptiveAvgPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool3d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            x = super().forward(x)

        elif self.step_mode == "m":
            if x.dim() != 6:
                raise ValueError(
                    f"expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!"
                )
            x = functional.seq_to_ann_forward(x, super().forward)

        return x
