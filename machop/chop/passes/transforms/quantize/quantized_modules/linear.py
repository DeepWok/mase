import math
from functools import partial
from math import ceil, log2

import torch
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
)


class _LinearBase(torch.nn.Linear):
    bypass = False
    _required_config_keys = None
    _optional_config_keys = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)
        else:
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

    def get_output_bitwidth(self) -> dict:
        raise NotImplementedError()


class LinearInteger(_LinearBase):
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
        assert config is not None, "config is None!"

        self.bypass = config.get("bypass", False)
        # establish quantizer
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
        self.config = config

    # def get_output_bitwidth(self):
    #     config = self.config
    #     w_width, w_frac = config["weight_width"], config["weight_frac_width"]
    #     x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
    #     bias_width = config["bias_width"]

    #     ops = self.in_features
    #     product_width = w_width + x_width
    #     product_frac_width = w_frac + x_frac
    #     # *: + 1 for bias
    #     output_width = max(bias_width, product_width + ceil(log2(ops))) + 1
    #     output_frac_width = product_frac_width

    #     o_bitwidth = {}
    #     o_bitwidth["data_out_width"] = output_width
    #     o_bitwidth["data_out_frac_width"] = output_frac_width
    #     # o_bitwidth["product_width"] = product_width
    #     # o_bitwidth["product_frac_width"] = product_frac_width
    #     return o_bitwidth


class LinearMinifloatDenorm(_LinearBase):
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

        self.config = config


class LinearMinifloatIEEE(_LinearBase):
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
        self.config = config

    # def get_output_bitwidth(self) -> dict:
    #     num_ops = self.in_features
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


class LinearLog(_LinearBase):
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
        self.config = config


class LinearBlockFP(_LinearBase):
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
        x_skip_first_dim = config.get("data_in_skip_first_dim", True)

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
            skip_first_dim=False,
        )
        self.x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_skip_first_dim,
        )
        self.b_quantizer = partial(
            block_fp_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias=b_exponent_bias,
            block_size=b_block_size,
            skip_first_dim=False,
        )

        self.config = config


class LinearBlockMinifloat(_LinearBase):
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
        x_skip_first_dim = config.get("data_in_skip_first_dim", True)

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
            skip_first_dim=False,
        )
        self.x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_skip_first_dim,
        )
        self.b_quantizer = partial(
            block_minifloat_quantizer,
            width=b_width,
            exponent_width=b_exponent_width,
            exponent_bias_width=b_exponent_bias_width,
            block_size=b_block_size,
            skip_first_dim=False,
        )

        self.config = config


class LinearBlockLog(_LinearBase):
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

        assert config is not None, "config is None!"

        self.bypass = config.get("bypass", False)
        # establish quantizers
        w_width, w_exponent_bias_width, w_block_size = (
            config["weight_width"],
            config["weight_exponent_bias_width"],
            config["weight_block_size"],
        )
        x_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        x_skip_first_dim = config.get("data_in_skip_first_dim", True)

        b_width, b_exponent_bias_width, b_block_size = (
            config["bias_width"],
            config["bias_exponent_bias_width"],
            config["bias_block_size"],
        )

        # blocking/unblocking 4D kernel/feature map is not supported
        self.w_quantizer = partial(
            block_log_quantizer,
            width=w_width,
            exponent_bias_width=w_exponent_bias_width,
            block_size=w_block_size,
            skip_first_dim=False,
        )
        self.x_quantizer = partial(
            block_log_quantizer,
            width=x_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_skip_first_dim,
        )
        self.b_quantizer = partial(
            block_log_quantizer,
            width=b_width,
            exponent_bias_width=b_exponent_bias_width,
            block_size=b_block_size,
            skip_first_dim=False,
        )
        self.config = config
