from functools import partial

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from chop.nn.quantizers import (
    integer_quantizer,
)


class _LayerNormBase(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        assert elementwise_affine == False, "elementwise_affine not supported!"
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.bypass:
            x = self.x_quantizer(x)
        return F.layer_norm(x, self.normalized_shape, None, None)


class LayerNormInteger(_LayerNormBase):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
