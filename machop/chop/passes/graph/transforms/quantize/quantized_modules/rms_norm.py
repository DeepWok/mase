from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..quantizers import (
    integer_quantizer,
)
import chop.models.manual.rms_norm as rms


class _RMSNormBase(rms.RMSNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.bypass = False
        self.x_quantizer = None
        self.w_quantizer = None

    def forward(self, x: Tensor):
        return rms._rms_norm(x, self.eps, self.weight)


class RMSNormInteger(_RMSNormBase):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = False,
        device=None,
        dtype=None,
        config=None,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        self.x_quantizer = partial(
            integer_quantizer,
            width=config["data_in_width"],
            frac_width=config["data_in_frac_width"],
        )
        self.w_quantizer = partial(
            integer_quantizer,
            width=config["weight_width"],
            frac_width=config["weight_frac_width"],
        )
