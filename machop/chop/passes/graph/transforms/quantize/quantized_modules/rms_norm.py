from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..quantizers import (
    integer_quantizer,
)


def _rms_norm(x: Tensor, eps, bias, scale, offset):
    mean_squares = x.square().mean(dim=(1, 2, 3), keepdim=True)
    rms_x = mean_squares.sqrt()
    x_normed = x / (rms_x + eps)
    if bias:
        return scale * x_normed + offset
    return scale * x_normed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.bias = bias

        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.elementwise_affine:
            self.scale = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            if bias:
                self.offset = nn.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('scale', None)
        else:
            self.register_parameter('scale', None)
            self.register_parameter('offset', None)

    def forward(self, x: Tensor):
        if self.elementwise_affine:
            offset = 0.0 if self.bias else self.offset
            return _rms_norm(x, self.eps, self.bias, self.scale, offset)
        return _rms_norm(x, self.eps, False, 1.0, 0.0)


class _RMSNormBase(RMSNorm):
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
            return _rms_norm(x, self.eps, self.bias, self.scale, offset)
        return _rms_norm(x, self.eps, False, 1.0, 0.0)


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
