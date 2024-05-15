from functools import partial

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from chop.nn.quantizers import (
    integer_quantizer,
    binary_quantizer,
)
from .utils import quantiser_passthrough


class _BatchNorm2dBase(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
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

        if not self.training or self.track_running_stats:
            running_mean = self.running_mean
            running_var = self.running_var
        else:
            running_mean = None
            running_var = None

        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean,
            running_var,
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

        if affine:
            w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
            b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
            self.w_quantizer = partial(
                integer_quantizer, width=w_width, frac_width=w_frac_width
            )
            self.b_quantizer = partial(
                integer_quantizer, width=b_width, frac_width=b_frac_width
            )


class BatchNorm2dBinary(_BatchNorm2dBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
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
        w_stochastic = config["weight_stochastic"]
        w_bipolar = config["weight_bipolar"]
        self.w_quantizer = partial(
            binary_quantizer, stochastic=w_stochastic, bipolar=w_bipolar
        )
        self.b_quantizer = quantiser_passthrough
        self.x_quantizer = quantiser_passthrough
