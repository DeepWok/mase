from functools import partial

import torch.nn.functional as F

from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import (
    integer_quantizer,
    minifloat_ieee_quantizer,
    minifloat_simple_quantizer,
)


@mark_as_leaf_func
def relu_integer(x, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.relu(x, inplace=inplace)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)

        return F.relu(x_quantizer(x), inplace=inplace)


def construct_essential_config_relu_integer(config):
    return {
        "bypass": config.get("bypass", False),
        "data_in_width": config["data_in_width"],
        "data_in_frac_width": config["data_in_frac_width"],
    }


def get_output_bitwidth_relu_integer(config):
    return {
        "data_out_width": config["data_in_width"],
        "data_out_frac_width": config["data_in_frac_width"],
    }


@mark_as_leaf_func
def relu_minifloat_simple(x, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.relu(x, inplace=inplace)
    else:
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )

        x_quantizer = partial(
            minifloat_simple_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )

        return F.relu(x_quantizer(x), inplace=inplace)


@mark_as_leaf_func
def relu_minifloat_ieee(x, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.relu(x, inplace=inplace)
    else:
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )

        x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )

        return F.relu(x_quantizer(x), inplace=inplace)
