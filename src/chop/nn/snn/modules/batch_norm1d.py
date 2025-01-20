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


class BatchNorm1d(nn.BatchNorm1d, base.StepModule):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        step_mode="s",
    ):
        """
        * :ref:`API in English <BatchNorm1d-en>`

        .. _BatchNorm1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm1d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)

        elif self.step_mode == "m":
            if x.dim() != 4 and x.dim() != 3:
                raise ValueError(
                    f"expected x with shape [T, N, C, L] or [T, N, C], but got x with shape {x.shape}!"
                )
            return functional.seq_to_ann_forward(x, super().forward)
