from functools import partial
from math import ceil, log2
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t

from ..quantizers import (
    block_fp_quantizer,
    integer_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
)


class _AvgPool2dBase(torch.nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.avg_pool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        x = self.x_quantizer(x)
        # Here we have the same problem as quantized conv2d
        # we assume the accumulation is lossless
        return F.avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    # def get_output_bitwidth(self) -> dict:
    #     raise NotImplementedError


class AvgPool2dInteger(_AvgPool2dBase):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        config=None,
    ) -> None:
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )

    # def get_output_bitwidth(self) -> dict:
    #     config = self.config

    #     x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
    #     num_add_operands = self.kernel_size[0] * self.kernel_size[1]
    #     output_width = x_width + ceil(log2(num_add_operands))
    #     output_frac_width = x_frac

    #     o_bitwidth = {}
    #     o_bitwidth["data_out_width"] = output_width
    #     o_bitwidth["data_out_frac_width"] = output_frac_width
    #     return o_bitwidth


class _AdaptiveAvgPool2dBase(torch.nn.AdaptiveAvgPool2d):
    """
    Refer to https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993?newreg=f2a34d7176564a5288717e984bdc21c7
    """

    bypass = False

    def __init__(self, output_size) -> None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        super().__init__(output_size)
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.adaptive_avg_pool2d(input=x, output_size=self.output_size)
        else:
            pool2d_kwargs = self._get_pool2d_kwargs(x.shape)
            f_padding = pool2d_kwargs.pop("f_padding")
            x = F.pad(x, f_padding)
            x = self.x_quantizer(x)
            return F.avg_pool2d(x, **pool2d_kwargs)

    def _get_pool2d_kwargs(self, x_shape):
        h_in_new = ceil(x_shape[-2] / self.output_size[0]) * self.output_size[0]
        w_in_new = ceil(x_shape[-1] / self.output_size[1]) * self.output_size[1]
        f_padding = (0, h_in_new - x_shape[-2], 0, w_in_new - x_shape[-1])
        stride = (h_in_new // self.output_size[0], w_in_new // self.output_size[1])
        kernel_size = (
            h_in_new - (self.output_size[0] - 1) * stride[0],
            w_in_new - (self.output_size[1] - 1) * stride[1],
        )
        return {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": 0,
            "f_padding": f_padding,
        }

    # def get_output_bitwidth(self) -> dict:
    #     raise NotImplementedError


class AdaptiveAvgPool2dInteger(_AdaptiveAvgPool2dBase):
    def __init__(self, output_size, config) -> None:
        super().__init__(output_size)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]

        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )

    # def get_output_bitwidth(self, x_shape) -> dict:
    #     config = self.config

    #     pool2d_kwargs = self._get_pool2d_kwargs(x_shape=x_shape)
    #     kernel_size = pool2d_kwargs["kernel_size"]

    #     x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
    #     num_add_operands = kernel_size[0] * kernel_size[1]
    #     output_width = x_width + ceil(log2(num_add_operands))
    #     output_frac_width = x_frac

    #     o_bitwidth = {}
    #     o_bitwidth["data_out_width"] = output_width
    #     o_bitwidth["data_out_frac_width"] = output_frac_width

    #     return o_bitwidth
