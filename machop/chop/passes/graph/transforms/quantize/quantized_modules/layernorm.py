from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import (
    integer_quantizer,
)

class LayerNormInteger(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None, config=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype=None)
        
        
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )
        self.config = config

        
    def forward(self, input: Tensor) -> Tensor:
        weight = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias)
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        
