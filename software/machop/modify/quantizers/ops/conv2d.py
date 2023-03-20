from functools import partial
from typing import Union

import torch
from torch import Tensor
from torch.nn.common_types import _size_2_t

from ....graph.mase_tracer import mark_as_leaf_module
from ..quantizers import integer_quantizer
from .utils import extract_required_config


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


@mark_as_leaf_module
class Conv2dInteger(Conv2dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_frac_width",
        "input_width",
        "input_frac_width",
    )
    _optional_config_keys = ("bypass", "bias_width", "bias_frac_width")

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
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["input_width"], config["input_frac_width"]
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config.get("bias_width", None), config.get(
            "bias_frac_width", None
        )
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        if b_width is None:
            self.b_quantizer = self.w_quantizer
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", None)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_frac_width"] = config.get(
            "bias_frac_widht", config["weight_frac_width"]
        )
        return r_config | o_config
