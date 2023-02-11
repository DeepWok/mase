import torch

from torch import Tensor
from torch.nn.common_types import _size_2_t

from typing import Union
from ..quantizers import integer_quantizer
from functools import partial


class Conv2dBase(torch.nn.Conv2d):

    def forward(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        # WARNING: this may have been simplified, we are assuming here the accumulation is lossless!
        # The addition size is in_channels * K * K
        return self._conv_forward(x, w, bias)
    
    def get_quantized_weight(self) -> Tensor:
        return self.w_quantizer(self.weight)

    def get_quantized_weights_with_inputs(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        y = self._conv_forward(x, w, bias)
        return {
            'x': x,
            'w': w,
            'bias': bias,
            'y': y,
        }


class Conv2dInteger(torch.nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None,
            config=None) -> None:
        super().__init__(
            in_features=in_channels, 
            out_features=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)

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
