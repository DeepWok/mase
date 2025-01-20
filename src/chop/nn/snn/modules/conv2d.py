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


class Conv2d(nn.Conv2d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        """
        * :ref:`API in English <Conv2d-en>`
        .. _Conv2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv2d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            x = super().forward(x)

        elif self.step_mode == "m":
            if x.dim() != 5:
                raise ValueError(
                    f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
                )
            x = functional.seq_to_ann_forward(x, super().forward)

        return x
