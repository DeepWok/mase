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


class Upsample(nn.Upsample, base.StepModule):
    def __init__(
        self,
        size: Optional[_size_any_t] = None,
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        step_mode: str = "s",
    ) -> None:
        """
        * :ref:`API in English <Upsample-en>`

        .. _Upsample-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Upsample` for other parameters' API
        """
        super().__init__(
            size, scale_factor, mode, align_corners, recompute_scale_factor
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor) -> Tensor:
        if self.step_mode == "s":
            x = super().forward(x)

        elif self.step_mode == "m":
            x = functional.seq_to_ann_forward(x, super().forward)

        return x
