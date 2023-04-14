import math
from functools import partial
from math import ceil, log2
from typing import Union

import torch
from torch import Tensor
from torch.nn.common_types import _size_1_t

from ....graph.mase_tracer import mark_as_leaf_module
from ..quantizers import (
    integer_quantizer,
    log_quantizer,
    minifloat_ieee_quantizer,
    minifloat_simple_quantizer,
    msfp_quantizer,
)
from .utils import extract_required_config


class Conv1dBase(torch.nn.Conv1d):
    bypass = False

    _required_config_keys = None
    _optional_config_keys = None

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

    def construct_essential_config(self, config):
        """construct configs for HW gen"""
        raise NotImplementedError()

    def get_output_bitwidth(self):
        """output bit width info for HW gen"""
        raise NotImplementedError()


@mark_as_leaf_module
class Conv1dInteger(Conv1dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_frac_width",
        "data_in_width",
        "data_in_frac_width",
    )
    _optional_config_keys = ("bypass", "bias_width", "bias_frac_width")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        config=None,
    ) -> None:
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
            dtype=dtype,
        )
        assert config is not None, "config is None!"

        self.bypass = config.get("bypass", False)
        # establish quantizers
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
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
        o_config["bypass"] = config.get("bypass", False)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_frac_width"] = config.get(
            "bias_frac_width", config["weight_frac_width"]
        )
        return r_config | o_config

    def get_output_bitwidth(self):
        config = self.config
        w_width, w_frac = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
        bias_width = config["bias_width"]

        ops = self.in_channels * self.kernel_size[0]
        product_width = w_width + x_width
        product_frac_width = w_frac + x_frac
        # *: +1 for bias
        output_width = max(bias_width, product_width + ceil(log2(ops))) + 1
        output_frac_width = product_frac_width

        o_bitwidth = {}
        o_bitwidth["data_out_width"] = output_width
        o_bitwidth["data_out_frac_width"] = output_frac_width
        # output_bitwidth_info["product_width"] = product_width
        # output_bitwidth_info["product_frac_width"] = product_frac_width
        return o_bitwidth


@mark_as_leaf_module
class Conv1dMinifloatSimple(Conv1dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_exponent_width",
        "weight_exponent_bias",
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = (
        "bypass",
        "bias_width",
        "bias_exponent_width",
        "bias_exponent_bias",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        config: dict = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        w_width, w_exponent_width, w_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
        )
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        b_width, b_exponent_width, b_exponent_bias = (
            config.get("bias_width", None),
            config.get("bias_exponent_width", None),
            config.get("bias_exponent_bias", None),
        )

        self.w_quantizer = partial(
            minifloat_simple_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias=w_exponent_bias,
        )

        self.x_quantizer = partial(
            minifloat_simple_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )

        if b_width is None or b_exponent_width is None or b_exponent_bias is None:
            self.b_quantizer = self.w_quantizer
        else:
            self.b_quantizer = partial(
                minifloat_simple_quantizer,
                width=b_width,
                exponent_width=b_exponent_width,
                exponent_bias=b_exponent_bias,
            )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_exponent_width"] = config.get(
            "bias_exponent_width", config["weight_exponent_width"]
        )
        o_config["bias_exponent_bias"] = config.get(
            "bias_exponent_bias", config["weight_exponent_bias"]
        )
        return r_config | o_config


@mark_as_leaf_module
class Conv1dLog(Conv1dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_exponent_bias",
        "data_in_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = (
        "bypass",
        "bias_width",
        "bias_exponent_bias",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        config: dict = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        w_width, w_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_bias"],
        )
        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        b_width, b_exponent_bias = (
            config.get("bias_width", None),
            config.get("bias_exponent_bias", None),
        )

        self.w_quantizer = partial(
            log_quantizer,
            width=w_width,
            exponent_bias=w_exponent_bias,
        )

        self.x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )

        if b_width is None or b_exponent_bias is None:
            self.b_quantizer = self.w_quantizer
        else:
            self.b_quantizer = partial(
                log_quantizer,
                width=b_width,
                exponent_bias=b_exponent_bias,
            )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_exponent_bias"] = config.get(
            "bias_exponent_bias", config["weight_exponent_bias"]
        )
        return r_config | o_config


@mark_as_leaf_module
class Conv1dMinifloatIEEE(Conv1dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_exponent_width",
        "weight_exponent_bias",
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = (
        "bypass",
        "bias_width",
        "bias_exponent_width",
        "bias_exponent_bias",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        config: dict = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        w_width, w_exponent_width, w_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
        )
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        b_width, b_exponent_width, b_exponent_bias = (
            config.get("bias_width", None),
            config.get("bias_exponent_width", None),
            config.get("bias_exponent_bias", None),
        )

        self.w_quantizer = partial(
            minifloat_ieee_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias=w_exponent_bias,
        )

        self.x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )

        if b_width is None or b_exponent_width is None or b_exponent_bias is None:
            self.b_quantizer = self.w_quantizer
        else:
            self.b_quantizer = partial(
                minifloat_ieee_quantizer,
                width=b_width,
                exponent_width=b_exponent_width,
                exponent_bias=b_exponent_bias,
            )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_exponent_width"] = config.get(
            "bias_exponent_width", config["weight_exponent_width"]
        )
        o_config["bias_exponent_bias"] = config.get(
            "bias_exponent_bias", config["weight_exponent_bias"]
        )
        return r_config | o_config


@mark_as_leaf_module
class Conv1dMSFP(Conv1dBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_exponent_width",
        "weight_exponent_bias",
        "weight_block_size",
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
        "data_in_block_size",
        "bias_width",
        "bias_exponent_width",
        "bias_exponent_bias",
        "bias_block_size",
    )
    _optional_config_keys = ("bypass",)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        config: dict = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        w_width, w_exponent_width, w_exponent_bias, w_block_size = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
            config["weight_block_size"],
        )
        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )
        b_width, b_exponent_width, b_exponent_bias, b_block_size = (
            config["bias_width"],
            config["bias_exponent_width"],
            config["bias_exponent_bias"],
            config["bias_block_size"],
        )

        self.w_quantizer = partial(
            msfp_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias=w_exponent_bias,
            block_size=w_block_size,
            skip_first_dim=True,
        )

        self.x_quantizer = partial(
            msfp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=True,
        )

        self.b_quantizer = partial(
            msfp_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias=b_exponent_bias,
            block_size=b_block_size,
            skip_first_dim=False,
        )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config

    # def get_output_bitwidth(self) -> dict:
    #     k = self.kernel_size
    #     ic, oc = self.in_channels, self.out_channels
    #     # number of operations per pixel
    #     num_ops = k * ic
    #     product_bitwidth = self.w_width + self.x_width
    #     product_frac = self.w_frac_width + self.x_frac_width

    #     addition_bitwidth = math.ceil(math.log(num_ops))
    #     output_bitwidth = product_bitwidth + addition_bitwidth
    #     return {
    #         "output_width": output_bitwidth,
    #         "output_frac_width": product_frac,
    #         "product_width": product_bitwidth,
    #         "product_frac_width": product_frac,
    #     }
