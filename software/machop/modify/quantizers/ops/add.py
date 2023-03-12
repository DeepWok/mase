from functools import partial

import torch

from ..quantizers import integer_quantizer


class AddInteger(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_bits, x_fraction_bits = config["input_bits"], config["input_fraction_bits"]
        self.x_quantizer = partial(integer_quantizer, bits=x_bits, bias=x_fraction_bits)
        self.config = config

    def forward(self, x, y):
        x = self.x_quantizer(x)
        y = self.x_quantizer(y)
        return x + y
