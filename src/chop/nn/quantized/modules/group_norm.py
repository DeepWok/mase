import logging
from math import ceil, log2
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from chop.nn.quantizers import (
    integer_quantizer,
)


from mase_components.scalar_operators.fixed.test.isqrt_sw import isqrt_sw2

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class _GroupNormBase(nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 0.00001,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)

        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.bypass:
            x = self.x_quantizer(x)
        return F.group_norm(x, self.num_groups, None, None, self.eps)


class GroupNormInteger(_GroupNormBase):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 0.00001,
        affine: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
