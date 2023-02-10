import torch

from torch import Tensor
from torch.nn import functional as F

from ..quantizers import integer_quantizer
from functools import partial


class LinearInteger(torch.nn.Linear):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = True,
            device=None, dtype=None, config=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if config is None:
            raise ValueError('config is None for IntegerLinear')

        # establish quantizers
        w_bits, w_bias = config['weight_bits'], config['weight_bias']
        x_bits, x_bias = config['input_bits'], config['input_bias']
        # check bias quantizer, if not, use weight quantizer
        b_bits, b_bias = config.get('bias_bits', None), config.get('bias_bias', None) 
        self.w_quantizer = partial(integer_quantizer, bits=w_bits, bias=w_bias)
        self.x_quantizer = partial(integer_quantizer, bits=x_bits, bias=x_bias)
        if b_bits is None:
            self.b_quantizer = self.w_quantizer
        self.b_quantizer = partial(integer_quantizer, bits=b_bits, bias=b_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        return F.linear(x, w, bias)
    
    def get_quantized_weight(self):
        return self.w_quantizer(self.weight)

    def get_quantized_weights_with_inputs(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        y = F.linear(x, w, bias)
        return {
            'x': x,
            'w': w,
            'bias': bias,
            'y': y,
        }

