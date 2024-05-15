from functools import partial

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from chop.nn.quantizers import (
    integer_quantizer,
)


class _InstanceNorm2dBase(nn.InstanceNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        assert affine == False, "elementwise_affine not supported!"
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.bypass:
            x = self.x_quantizer(x)
        return F.instance_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class InstanceNorm2dInteger(_InstanceNorm2dBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
