from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import integer_quantizer, msfp_quantizer


class LinearBase(torch.nn.Linear):
    bypass = False

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        return F.linear(x, w, bias)

    def get_quantized_weight(self) -> Tensor:
        return self.w_quantizer(self.weight)

    def get_quantized_weights_with_inputs(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        y = F.linear(x, w, bias)
        return {
            "x": x,
            "w": w,
            "bias": bias,
            "y": y,
        }


class LinearInteger(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if config is None:
            raise ValueError("config is None for IntegerLinear")

        self.bypass = config.get("bypass", False)
        # establish quantizer
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


class LinearMSFP(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if config is None:
            raise ValueError("config is None for IntegerLinear")

        self.bypass = config.get("bypass", False)
        # establish quantizers
        w_bits, w_block_size, w_exponent_bits = (
            config["weight_bits"],
            config["weight_block_size"],
            config["weight_exponent_bits"],
        )
        x_bits, x_block_size, x_exponent_bits = (
            config["input_bits"],
            config["input_block_size"],
            config["input_exponent_bits"],
        )
        # check bias quantizer, if not, use weight quantizer
        b_bits, b_fraction_bits = config.get("bias_bits", None), config.get(
            "bias_fraction_bits", None
        )
        self.w_quantizer = partial(
            msfp_quantizer,
            bits=w_bits,
            exponent_bits=w_exponent_bits,
            block_size=w_block_size,
        )
        self.x_quantizer = partial(
            msfp_quantizer,
            bits=x_bits,
            exponent_bits=x_exponent_bits,
            block_size=x_block_size,
        )
        if b_bits is None:
            self.b_quantizer = self.w_quantizer
        self.b_quantizer = partial(integer_quantizer, bits=b_bits, bias=b_fraction_bits)
        self.config = config
