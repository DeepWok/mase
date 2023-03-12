from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import integer_quantizer


class ReLUInteger(torch.nn.ReLU):
    bypass = False

    def __init__(self, inplace: bool = False, config: dict = {}):
        super().__init__(inplace)
        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_bits, x_fraction_bias = config["input_bits"], config["input_fraction_bits"]
        self.x_quantizer = partial(
            integer_quantizer, bits=x_bits, fraction_bits=x_fraction_bias
        )
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        x = self.x_quantizer(x)
        return F.relu(x)
