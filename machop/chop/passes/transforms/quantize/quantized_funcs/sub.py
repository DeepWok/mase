from functools import partial

import torch

# TODO: Previoulsy we use tracer to mark custom ops, 
# which are now already defined in MaseGraph. 
# This process should be decrepated in quantizer
# from ....graph.mase_tracer import mark_as_leaf_func

from ..quantizers import (
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_ieee_quantizer,
    minifloat_simple_quantizer,
    block_fp_quantizer,
)


# @mark_as_leaf_func
def sub_integer(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
    else:
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)
        x = x_quantizer(x)
        y = x_quantizer(y)
        return x - y


def construct_essential_config_sub_integer(config):
    return {
        "bypass": config.get("bypass", False),
        "name": config["name"],
        "data_in_width": config["data_in_width"],
        "data_in_frac_width": config["data_in_frac_width"],
    }


def get_output_bitwidth_sub_integer(config):
    return {
        "data_out_width": config["data_in_width"] + 1,
        "data_out_frac_width": config["data_in_frac_width"],
    }


# @mark_as_leaf_func
def sub_minifloat_simple(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
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
        return x - y


# @mark_as_leaf_func
def sub_minifloat_ieee(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
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
        return x - y


# @mark_as_leaf_func
def sub_log(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
    else:
        x_width, x_exponent_bias = (
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        x_quantizer = partial(
            log_quantizer, width=x_width, exponent_bias=x_exponent_bias
        )
        x = x_quantizer(x)
        y = x_quantizer(y)
        return x - y


# @mark_as_leaf_func
def sub_block_fp(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
    else:
        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        # a hack to use 2d blocking
        if x.shape != y.shape:
            x, y = torch.broadcast_tensors(x, y)
        # assert x.shape == y.shape
        x_shape = [i for i in x.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        y = x_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, x_shape)
        return x - y


# @mark_as_leaf_func
def sub_block_minifloat(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
    else:
        x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        # a hack to use 2d blocking
        if x.shape != y.shape:
            x, y = torch.broadcast_tensors(x, y)
        # assert x.shape == y.shape
        x_shape = [i for i in x.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        y = x_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, x_shape)
        return x - y


# @mark_as_leaf_func
def sub_block_minifloat(x, y, config):
    bypass = config.get("bypass", False)
    if bypass:
        return x - y
