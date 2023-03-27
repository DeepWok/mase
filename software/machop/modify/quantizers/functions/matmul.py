from functools import partial

import torch

from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import (
    integer_quantizer,
    minifloat_ieee_quantizer,
    minifloat_simple_quantizer,
)

# PyTorch has torch.matmul and torch.bmm for matrix multiplication
matmul_mapping = {"matmul": torch.matmul, "bmm": torch.bmm}


def generic_matmul_integer(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)
        y_width, y_frac_width = config["weight_width"], config["weight_frac_width"]
        y_quantizer = partial(integer_quantizer, width=y_width, frac_width=y_frac_width)

        x = x_quantizer(x)
        y = y_quantizer(y)

        return matmul(x, y)


def generic_matmul_minifloat_simple(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        y_width, y_exponent_width, y_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
        )

        x_quantizer = partial(
            minifloat_simple_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        y_quantizer = partial(
            minifloat_simple_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias=y_exponent_bias,
        )
        x = x_quantizer(x)
        y = y_quantizer(y)
        return matmul(x, y)


def generic_matmul_minifloat_ieee(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        y_width, y_exponent_width, y_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
        )

        x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        y_quantizer = partial(
            minifloat_ieee_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias=y_exponent_bias,
        )
        x = x_quantizer(x)
        y = y_quantizer(y)
        return matmul(x, y)


# --------------------------------------------
@mark_as_leaf_func
def matmul_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "matmul")


@mark_as_leaf_func
def matmul_minifloat_simple(x, y, config):
    return generic_matmul_minifloat_simple(x, y, config, "matmul")


@mark_as_leaf_func
def matmul_minifloat_ieee(x, y, config):
    return generic_matmul_minifloat_ieee(x, y, config, "matmul")


@mark_as_leaf_func
def bmm_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "bmm")


@mark_as_leaf_func
def bmm_minifloat_simple(x, y, config):
    return generic_matmul_minifloat_simple(x, y, config, "bmm")


@mark_as_leaf_func
def bmm_minifloat_ieee(x, y, config):
    return generic_matmul_minifloat_ieee(x, y, config, "bmm")
