from functools import partial
from typing import Union

import torch
from torch import Tensor
from torch.nn.common_types import _size_2_t

from ..quantizers import integer_quantizer


class Conv2dBase(torch.nn.Conv2d):
    bypass = False

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self._conv_forward(x, self.weight, self.bias)
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
            "x": x,
            "w": w,
            "bias": bias,
            "y": y,
        }


class Conv2dInteger(Conv2dBase):
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
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.bypass = config.get("bypass", False)
        # establish quantizers
        w_bits, w_fraction_bits = config["weight_bits"], config["weight_fraction_bits"]
        x_bits, x_fraction_bits = config["input_bits"], config["input_fraction_bits"]
        # check bias quantizer, if not, use weight quantizer
        b_bits, b_fraction_bits = config.get("bias_bits", None), config.get(
            "bias_fraction_bits", None
        )
        self.w_quantizer = partial(
            integer_quantizer, bits=w_bits, fraction_bits=w_fraction_bits
        )
        self.x_quantizer = partial(
            integer_quantizer, bits=x_bits, fraction_bits=x_fraction_bits
        )
        if b_bits is None:
            self.b_quantizer = self.w_quantizer
        self.b_quantizer = partial(
            integer_quantizer, bits=b_bits, fraction_bits=b_fraction_bits
        )
        self.config = config
