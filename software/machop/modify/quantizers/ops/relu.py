import torch

from torch import Tensor
from torch.nn import functional as F

from ..quantizers import integer_quantizer
from functools import partial


class ReLUInteger(torch.nn.ReLU):

    def __init__(self, inplace: bool = False, config: dict = {}):
        super().__init__(inplace)

        # establish quantizers
        x_bits, x_bias = config['input_bits'], config['input_bias']
        self.x_quantizer = partial(integer_quantizer, bits=x_bits, bias=x_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        return F.relu(x)
