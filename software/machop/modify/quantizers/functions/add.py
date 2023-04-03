from functools import partial

from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import (
    integer_quantizer,
    minifloat_ieee_quantizer,
    minifloat_simple_quantizer,
)


@mark_as_leaf_func
def add_integer(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x + y
    else:
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)
        x = x_quantizer(x)
        y = x_quantizer(y)
        return x + y


def construct_essential_config_add_integer(config):
    return {
        "bypass": config.get("bypass", False),
        "data_in_width": config["data_in_width"],
        "data_in_frac_width": config["data_in_frac_width"],
    }


def get_output_bitwidth_add_integer(config):
    return {
        "data_out_width": config["data_in_width"] + 1,
        "data_out_frac_width": config["data_in_frac_width"],
    }


@mark_as_leaf_func
def add_minifloat_simple(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x + y
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

        x = x_quantizer(x)
        y = x_quantizer(y)
        return x + y


@mark_as_leaf_func
def add_minifloat_ieee(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x + y
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

        x = x_quantizer(x)
        y = x_quantizer(y)
        return x + y
