from functools import partial

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..quantizers import (
    integer_quantizer,
)
from .group_norm2d import _fixed_group_norm_2d_model


class _LayerNormBase(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
    ) -> None:

        assert elementwise_affine == False, "elementwise_affine not supported!"
        super().__init__(
            normalized_shape, eps, elementwise_affine, bias, device, dtype
        )
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.bypass:
            x = self.x_quantizer(x)
        return F.layer_norm(
            x, self.normalized_shape, None, None, 0  # TODO: Change eps
        )


class LayerNormInteger(_LayerNormBase):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
        config = None,
    ) -> None:
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


class LayerNormHWInteger(nn.Module):
    def __init__(
        self,
        normalized_shape,
        config = None,
    ) -> None:
        super().__init__()
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.layer_norm(x, self.normalized_shape)
        else:
            x_float, x_int = _fixed_group_norm_2d_model(
                x=x,
                in_width=self.config["data_in_width"],
                in_frac_width=self.config["data_in_frac_width"],
                variance_width=self.config["variance_width"],
                variance_frac_width=self.config["variance_frac_width"],
                inv_sqrt_width=self.config["inv_sqrt_width"],
                inv_sqrt_frac_width=self.config["inv_sqrt_frac_width"],
                out_width=self.config["out_width"],
                out_frac_width=self.config["out_frac_width"],
            )
            return x_float
