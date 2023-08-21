from functools import partial
from math import ceil, log2
from typing import Union, Optional

from .utils import get_stats

import torch
from chop.passes.transforms.quantize.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
)
from torch import Tensor
from torch.nn.common_types import _size_2_t

# LUTNet
import numpy as np
from typing import Type
from ..quantizers.LUTNet.BaseTrainer import BaseTrainer, LagrangeTrainer
from ..quantizers.LUTNet.MaskBase import MaskBase, MaskExpanded
import math


class _Conv2dBase(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
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
        self.bypass = False
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None

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

    def get_quantized_weights_with_inputs(self, x: Tensor) -> dict:
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

    def get_output_bitwidth(self) -> dict:
        raise NotImplementedError()


class Conv2dInteger(_Conv2dBase):
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
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        # establish quantizers
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )

    # def get_output_bitwidth(self) -> dict:
    #     config = self.config

    #     w_width, w_frac = config["weight_width"], config["weight_frac_width"]
    #     x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
    #     bias_width = config["bias_width"]

    #     ops = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
    #     product_width = w_width + x_width
    #     product_frac_width = w_frac + x_frac
    #     # *: +1 for bias
    #     output_width = max(bias_width, product_width + ceil(log2(ops))) + 1
    #     output_frac_width = product_frac_width

    #     o_bitwidth = {}
    #     o_bitwidth["data_out_width"] = output_width
    #     o_bitwidth["data_out_frac_width"] = output_frac_width
    #     # o_bitwidth["product_width"] = product_width
    #     # o_bitwidth["product_frac_width"] = product_frac_width
    #     return o_bitwidth


class Conv2dMinifloatDenorm(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

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
            config["bias_width"],
            config["bias_exponent_width"],
            config["bias_exponent_bias"],
        )

        self.w_quantizer = partial(
            minifloat_denorm_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias=w_exponent_bias,
        )

        self.x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )

        self.b_quantizer = partial(
            minifloat_denorm_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias=b_exponent_bias,
        )


class Conv2dMinifloatIEEE(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

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
            config["bias_width"],
            config["bias_exponent_width"],
            config["bias_exponent_bias"],
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

        self.b_quantizer = partial(
            minifloat_ieee_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias=b_exponent_bias,
        )


class Conv2dLog(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_width, w_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_bias"],
        )
        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        b_width, b_exponent_bias = (
            config["bias_width"],
            config["bias_exponent_bias"],
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

        self.b_quantizer = partial(
            log_quantizer,
            width=b_width,
            exponent_bias=b_exponent_bias,
        )


class Conv2dLog(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_width, w_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_bias"],
        )
        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        b_width, b_exponent_bias = (
            config["bias_width"],
            config["bias_exponent_bias"],
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

        self.b_quantizer = partial(
            log_quantizer,
            width=b_width,
            exponent_bias=b_exponent_bias,
        )


class Conv2dBlockFP(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

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

        # blocking/unblocking 4D kernel/feature map is not supported
        self.w_quantizer = partial(
            block_fp_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias=w_exponent_bias,
            block_size=w_block_size,
            skip_first_dim=True,
        )
        self.x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=True,
        )
        self.b_quantizer = partial(
            block_fp_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias=b_exponent_bias,
            block_size=b_block_size,
            skip_first_dim=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self._conv_forward(x, self.weight, self.bias)
        x_shape = [i for i in x.shape]
        w_shape = [i for i in self.weight.shape]
        # a hack for handling 4D block/unblock
        x = torch.flatten(x, 0, 1)
        x = self.x_quantizer(x)
        x = torch.reshape(x, x_shape)
        w = torch.flatten(self.weight, 0, 1)
        w = self.w_quantizer(w)
        w = torch.reshape(w, w_shape)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        # WARNING: this may have been simplified, we are assuming here the accumulation is lossless!
        # The addition size is in_channels * K * K
        return self._conv_forward(x, w, bias)


class Conv2dBlockMinifloat(_Conv2dBase):
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_width, w_exponent_width, w_exponent_bias_width, w_block_size = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias_width"],
            config["weight_block_size"],
        )
        x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        b_width, b_exponent_width, b_exponent_bias_width, b_block_size = (
            config["bias_width"],
            config["bias_exponent_width"],
            config["bias_exponent_bias_width"],
            config["bias_block_size"],
        )

        # blocking/unblocking 4D kernel/feature map is not supported
        self.w_quantizer = partial(
            block_minifloat_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            exponent_bias_width=w_exponent_bias_width,
            block_size=w_block_size,
            skip_first_dim=True,
        )
        self.x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=True,
        )
        self.b_quantizer = partial(
            block_minifloat_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias_width=b_exponent_bias_width,
            block_size=b_block_size,
            skip_first_dim=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self._conv_forward(x, self.weight, self.bias)
        x_shape = [i for i in x.shape]
        w_shape = [i for i in self.weight.shape]
        x = torch.flatten(x, 0, 1)
        x = self.x_quantizer(x)
        x = torch.reshape(x, x_shape)
        w = torch.flatten(self.weight, 0, 1)
        w = self.w_quantizer(w)
        w = torch.reshape(w, w_shape)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        # WARNING: this may have been simplified, we are assuming here the accumulation is lossless!
        # The addition size is in_channels * K * K
        return self._conv_forward(x, w, bias)


class Conv2dBlockLog(_Conv2dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
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
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_width, w_exponent_bias_width, block_size = (
            config["weight_width"],
            config["weight_exponent_bias_width"],
            config["weight_block_size"],
        )
        x_width, x_exponent_bias_width, block_size = (
            config["data_in_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        b_width, b_exponent_bias_width, block_size = (
            config["bias_width"],
            config["bias_exponent_bias_width"],
            config["bias_block_size"],
        )

        # blocking/unblocking 4D kernel/feature map is not supported
        self.w_quantizer = partial(
            block_log_quantizer,
            width=w_width,
            exponent_bias_width=w_exponent_bias_width,
            block_size=block_size,
            skip_first_dim=True,
        )
        self.x_quantizer = partial(
            block_log_quantizer,
            width=x_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=block_size,
            skip_first_dim=True,
        )
        self.b_quantizer = partial(
            block_log_quantizer,
            width=b_width,
            exponent_bias_width=b_exponent_bias_width,
            block_size=block_size,
            skip_first_dim=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self._conv_forward(x, self.weight, self.bias)
        x_shape = [i for i in x.shape]
        w_shape = [i for i in self.weight.shape]
        x = torch.flatten(x, 0, 1)
        x = self.x_quantizer(x)
        x = torch.reshape(x, x_shape)
        w = torch.flatten(self.weight, 0, 1)
        w = self.w_quantizer(w)
        w = torch.reshape(w, w_shape)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        # WARNING: this may have been simplified, we are assuming here the accumulation is lossless!
        # The addition size is in_channels * K * K
        return self._conv_forward(x, w, bias)


class Conv2dBinary(_Conv2dBase):
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
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        # establish quantizers
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_stochastic, b_stochastic, w_stochastic = (
            config["data_in_stochastic"],
            config["bias_stochastic"],
            config["weight_stochastic"],
        )
        x_bipolar, b_bipolar, w_bipolar = (
            config["data_in_bipolar"],
            config["bias_bipolar"],
            config["weight_bipolar"],
        )

        self.w_quantizer = partial(
            binary_quantizer, stochastic=w_stochastic, bipolar=w_bipolar
        )
        self.x_quantizer = partial(
            binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar
        )
        self.b_quantizer = partial(
            binary_quantizer, stochastic=b_stochastic, bipolar=b_bipolar
        )


class Conv2dTernary(_Conv2dBase):
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
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        # establish quantizers
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_scaling_factor = config["data_in_scaling_factor"]
        w_scaling_factor = config["weight_scaling_factor"]
        b_scaling_factor = config["bias_scaling_factor"]
        x_mean = get_stats(config, "data_in_mean")
        x_median = get_stats(config, "data_in_median")
        x_max = get_stats(config, "data_in_max")
        w_mean = get_stats(config, "weight_mean")
        w_median = get_stats(config, "weight_median")
        w_max = get_stats(config, "weight_max")
        b_mean = get_stats(config, "bias_mean")
        b_median = get_stats(config, "bias_median")
        b_max = get_stats(config, "bias_max")
        self.x_quantizer = partial(
            ternary_quantizer,
            scaling_factor=x_scaling_factor,
            maximum=x_max,
            median=x_median,
            mean=x_mean,
        )
        self.w_quantizer = partial(
            ternary_quantizer,
            scaling_factor=w_scaling_factor,
            maximum=w_max,
            median=w_median,
            mean=w_mean,
        )
        self.b_quantizer = partial(
            ternary_quantizer,
            scaling_factor=b_scaling_factor,
            maximum=b_max,
            median=b_median,
            mean=b_mean,
        )


class Conv2dLUT(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    padding: bool
    dilation: tuple
    groups: tuple
    bias: torch.Tensor
    padding_mode: str
    input_dim: tuple
    device: Optional[str]
    input_mask: torch.Tensor
    k: int
    trainer: BaseTrainer
    mask_builder_type: Type[MaskBase]
    mask_builder: MaskBase
    tables_count: int

    def __init__(
        self,
        config: None,  # mase configuration
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: Union[int, tuple] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        trainer_type: Type[BaseTrainer] = LagrangeTrainer,
        mask_builder_type: Type[MaskBase] = MaskExpanded,
        # k: int = 2,
        # binarization_level: int = 0,
        # input_expanded: bool = True,
        # input_dim: Union[int, tuple] = None,
        device: str = None,
    ):
        super(Conv2dLUT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)

        self.mask_builder_type = mask_builder_type

        self.groups = torch.nn.modules.utils._pair(groups)
        self.bias = None
        self.padding_mode = padding_mode
        self.k = config["data_in_k"]
        self.input_dim = torch.nn.modules.utils._pair(config["data_in_dim"])
        self.device = device
        self.input_mask = self._input_mask_builder()
        self.tables_count = self.mask_builder.get_tables_count()
        self.trainer = trainer_type(
            tables_count=self.tables_count,
            k=config["data_in_k"],
            binarization_level=(  # binarization_level 1 is binarized weight, 0 is not binarized
                1 if config["data_in_binarization_level"] == 1 else 0
            ),
            input_expanded=config["data_in_input_expanded"],
            device=device,
        )

        self.unfold = torch.nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        self.fold = torch.nn.Fold(
            output_size=(self._out_dim(0), self._out_dim(1)),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

    def _get_kernel_selections(self, channel_id):
        result = []
        for kh_index in range(self.kernel_size[0]):
            for kw_index in range(self.kernel_size[1]):
                result.append((channel_id, kh_index, kw_index))
        return set(result)

    def _table_input_selections(self):
        result = []
        for out_index in range(self.out_channels):
            for input_index in range(self.in_channels):
                selections = self._get_kernel_selections(
                    input_index
                )  # [(channel_id, kh, kw)...] 9
                for kh_index in range(self.kernel_size[0]):
                    for kw_index in range(self.kernel_size[1]):
                        conv_index = (input_index, kh_index, kw_index)
                        sub_selections = list(selections - set([conv_index]))
                        result.append(
                            (conv_index, sub_selections)
                        )  # [(channel_id, kh, kw),[(channel_id, kh, kw)...]] 9
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        result = []
        selections = self._table_input_selections()  # [kw * kh * ic * oc]
        self.mask_builder = self.mask_builder_type(self.k, selections, True)
        result.append(self.mask_builder.build())
        return np.concatenate(result)

    def _out_dim(self, dim):
        _out = (
            self.input_dim[dim]
            + 2 * self.padding[dim]
            - self.dilation[dim] * (self.kernel_size[dim] - 1)
            - 1
        ) / self.stride[dim]
        return math.floor(_out + 1)

    def forward(
        self,
        input: torch.Tensor,  # [10, 256, 3, 3]
        targets: torch.tensor = None,
        initalize: bool = False,
    ):
        assert len(input.shape) == 4
        batch_size = input.shape[0]
        folded_input = self.unfold(input).transpose(1, 2)  # [10, 1, 2304]
        folded_input = folded_input.view(
            batch_size,
            -1,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )  # [10, 1, 256, 3, 3]
        # print(self.input_mask.shape) # [1179648, 3] 256*256*9*2 NOTE: each element in the kernal corespond a table
        expanded_input = folded_input[
            :,
            :,
            self.input_mask[:, 0],
            self.input_mask[:, 1],
            self.input_mask[:, 2],
        ]  # [10, 1, 1179648]
        output = self.trainer(
            expanded_input, targets, initalize
        ).squeeze()  # [10, 589824]
        output = output.view(
            batch_size,
            self.out_channels,
            self._out_dim(0),
            self._out_dim(1),
            -1,
        ).sum(
            -1
        )  # [10, 256, 1, 1, 2304] -> [10, 256, 1, 1]
        output = output.view(
            batch_size, self._out_dim(0) * self._out_dim(1), -1
        ).transpose(
            1, 2
        )  # [10, 1, 256] -> [10, 256, 1]
        output = output.view(
            batch_size, self.out_channels, self._out_dim(0), self._out_dim(1)
        )  # [10, 256, 1, 1]

        return output

    def pre_initialize(self):
        self.trainer.clear_initializion()

    def update_initialized_weights(self):
        self.trainer.update_initialized_weights()
