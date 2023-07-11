from functools import partial
from math import ceil, log2

import torch

# from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
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
        # y = x_quantizer(y)

        return matmul(x, y)


def generic_matmul_binary(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_stochastic = config["stochastic"]
        x_bipolar = config["bipolar"]
        x_quantizer = partial(
            binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar
        )
        x = x_quantizer(x)
        y = x_quantizer(y)
        # y = x_quantizer(y)

        return matmul(x, y)


# def construct_essential_config_generic_matmul_integer(config):
#     return {
#         "bypass": config.get("bypass", False),
#         "name": config["name"],
#         "data_in_width": config["data_in_width"],
#         "data_in_frac_width": config["data_in_frac_width"],
#         "weight_width": config["weight_width"],
#         "weight_frac_width": config["weight_frac_width"],
#     }


def generic_matmul_minifloat_denorm(x, y, config, style="matmul"):
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
            minifloat_denorm_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        y_quantizer = partial(
            minifloat_denorm_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias=y_exponent_bias,
        )
        x = x_quantizer(x)
        y = y_quantizer(y)
        # y = x_quantizer(y)
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


def generic_matmul_log(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        y_width, y_exponent_bias = (
            config["weight_width"],
            config["weight_exponent_bias"],
        )

        x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )
        y_quantizer = partial(
            log_quantizer,
            width=y_width,
            exponent_bias=y_exponent_bias,
        )
        x = x_quantizer(x)
        y = y_quantizer(y)
        # y = x_quantizer(y)
        return matmul(x, y)


def generic_matmul_block_fp(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )
        y_width, y_exponent_width, y_exponent_bias, y_block_size = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
            config["weight_block_size"],
        )
        x_more_than_2_dims = x.ndim > 2
        y_more_than_2_dims = y.ndim > 2

        x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        y_quantizer = partial(
            block_fp_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias=y_exponent_bias,
            block_size=y_block_size,
            skip_first_dim=y_more_than_2_dims,
        )
        # flatten all other dims except for the last two dims for performing matmul
        # this is a hack for allowing block/unblock the last two dims of multiple dim tensors
        x_shape = [i for i in x.shape]
        y_shape = [i for i in y.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
        if y_more_than_2_dims:
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        # y = x_quantizer(y)
        y = y_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, y_shape)
        return matmul(x, y)


def generic_matmul_block_minifloat(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        y_width, y_exponent_width, y_exponent_bias_width, y_block_size = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias_width"],
            config["weight_block_size"],
        )
        x_more_than_2_dims = x.ndim > 2
        y_more_than_2_dims = y.ndim > 2

        x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        y_quantizer = partial(
            block_minifloat_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias_width=y_exponent_bias_width,
            block_size=y_block_size,
            skip_first_dim=y_more_than_2_dims,
        )
        # flatten all other dims except for the last two dims for performing matmul
        # this is a hack for allowing block/unblock the last two dims of multiple dim tensors
        x_shape = [i for i in x.shape]
        y_shape = [i for i in y.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
        if y_more_than_2_dims:
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        # y = x_quantizer(y)
        y = y_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, y_shape)
        return matmul(x, y)


def generic_matmul_block_log(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        y_width, y_exponent_bias_width, y_block_size = (
            config["weight_width"],
            config["weight_exponent_bias_width"],
            config["weight_block_size"],
        )
        x_more_than_2_dims = x.ndim > 2
        y_more_than_2_dims = y.ndim > 2

        x_quantizer = partial(
            block_log_quantizer,
            width=x_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        y_quantizer = partial(
            block_log_quantizer,
            width=y_width,
            exponent_bias_width=y_exponent_bias_width,
            block_size=y_block_size,
            skip_first_dim=y_more_than_2_dims,
        )
        # flatten all other dims except for the last two dims for performing matmul
        # this is a hack for allowing block/unblock the last two dims of multiple dim tensors
        x_shape = [i for i in x.shape]
        y_shape = [i for i in y.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
        if y_more_than_2_dims:
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        # y = x_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, y_shape)
        return matmul(x, y)


def matmul_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "matmul")


def matmul_binary(x, y, config):
    return generic_matmul_binary(x, y, config, "matmul")


def matmul_minifloat_denorm(x, y, config):
    return generic_matmul_minifloat_denorm(x, y, config, "matmul")


def matmul_minifloat_ieee(x, y, config):
    return generic_matmul_minifloat_ieee(x, y, config, "matmul")


def matmul_log(x, y, config):
    return generic_matmul_log(x, y, config, "matmul")


def matmul_block_fp(x, y, config):
    return generic_matmul_block_fp(x, y, config, "matmul")


def matmul_block_minifloat(x, y, config):
    return generic_matmul_block_minifloat(x, y, config, "matmul")


def matmul_block_log(x, y, config):
    return generic_matmul_block_log(x, y, config, "matmul")


def bmm_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "bmm")


def bmm_binary(x, y, config):
    return generic_matmul_binary(x, y, config, "bmm")


# def get_output_bitwidth_bmm_integer(config, x_shape):
#     w_width, w_frac = config["weight_width"], config["weight_frac_width"]
#     x_width, x_frac = config["data_in_width"], config["data_in_frac_width"]
#     ops = x_shape[-1]
#     product_width = w_width + x_width
#     product_frac_width = w_frac + x_frac
#     output_width = product_width + ceil(log2(ops))
#     output_frac_width = product_frac_width

#     o_bitwidth = {}
#     o_bitwidth["data_out_width"] = output_width
#     o_bitwidth["data_out_frac_width"] = output_frac_width
#     return o_bitwidth


def bmm_minifloat_denorm(x, y, config):
    return generic_matmul_minifloat_denorm(x, y, config, "bmm")


def bmm_minifloat_ieee(x, y, config):
    return generic_matmul_minifloat_ieee(x, y, config, "bmm")


def bmm_log(x, y, config):
    return generic_matmul_log(x, y, config, "bmm")


def bmm_block_fp(x, y, config):
    return generic_matmul_block_fp(x, y, config, style="bmm")


def bmm_block_minifloat(x, y, config):
    return generic_matmul_block_minifloat(x, y, config, style="bmm")


def bmm_block_log(x, y, config):
    return generic_matmul_block_log(x, y, config, style="bmm")
