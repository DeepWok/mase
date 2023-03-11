import torch

from functools import partial
from ..quantizers import integer_quantizer


class AddInteger(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_bits, x_bias = config["input_bits"], config["input_bias"]
        self.x_quantizer = partial(integer_quantizer, bits=x_bits, bias=x_bias)
        self.config = config

    def forward(self, x, y):
        x = self.x_quantizer(x)
        y = self.x_quantizer(y)
        return x + y
