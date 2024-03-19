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
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
    ):
        # Hardware constraints
        assert elementwise_affine == False, "Not implemented."
        assert bias == False, "Not implemented."

        super().__init__(
            normalized_shape, eps, elementwise_affine, bias, device, dtype
        )
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor):
        if not self.bypass:
            x = self.x_quantizer(x)
        if self.elementwise_affine:
            offset = 0.0 if self.bias else self.offset
            return rms._rms_norm(x, self.eps, self.bias, self.scale, offset)
        return rms._rms_norm(x, self.eps, False, 1.0, 0.0)


class RMSNormInteger(_RMSNormBase):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
        config = None,
    ):
        super().__init__(
            normalized_shape, eps, elementwise_affine, bias, device, dtype
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
