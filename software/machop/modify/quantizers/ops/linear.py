from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ....graph.mase_tracer import mark_as_leaf_module
from ..quantizers import integer_quantizer, msfp_quantizer
from .utils import extract_required_config


@mark_as_leaf_module
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


@mark_as_leaf_module
class LinearInteger(LinearBase):
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
        self.config = config

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        o_config["bias_width"] = config.get("bias_width", config["weight_width"])
        o_config["bias_frac_width"] = config.get(
            "bias_frac_width", config["weight_frac_width"]
        )
        return r_config | o_config


@mark_as_leaf_module
class LinearMSFP(LinearBase):
    _required_config_keys = (
        "name",
        "weight_width",
        "weight_block_size",
        "weight_exponent_width",
        "input_width",
        "input_block_size",
        "input_exponent_width",
    )
    _optional_config_keys = ("bypass", "bias_width", "bias_frac_width")

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
        w_width, w_block_size, w_exponent_width = (
            config["weight_width"],
            config["weight_block_size"],
            config["weight_exponent_width"],
        )
        x_width, x_block_size, x_exponent_width = (
            config["input_width"],
            config["input_block_size"],
            config["input_exponent_width"],
        )
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config.get("bias_width", None), config.get(
            "bias_frac_width", None
        )
        self.w_quantizer = partial(
            msfp_quantizer,
            width=w_width,
            exponent_width=w_exponent_width,
            block_size=w_block_size,
        )
        self.x_quantizer = partial(
            msfp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            block_size=x_block_size,
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
