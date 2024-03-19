from functools import partial

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..quantizers import (
    integer_quantizer,
)


class _BatchNorm2dBase(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device = None,
        dtype = None,
    ) -> None:

        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if not self.bypass:
            x = self.x_quantizer(x)

        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            self.momentum,
            self.eps,
        )


class BatchNorm2dInteger(_BatchNorm2dBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device = None,
        dtype = None,
        config = None,
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
