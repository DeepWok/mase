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


class GroupNorm(nn.GroupNorm, base.StepModule):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        step_mode="s",
    ):
        """
        * :ref:`API in English <GroupNorm-en>`

        .. _GroupNorm-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.GroupNorm` for other parameters' API
        """
        super().__init__(num_groups, num_channels, eps, affine)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)

        elif self.step_mode == "m":
            return functional.seq_to_ann_forward(x, super().forward)
