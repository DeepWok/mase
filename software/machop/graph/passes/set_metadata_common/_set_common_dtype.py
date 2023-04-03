"""
hook function for analyzing dtype & quantization info

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["type"&"precision"&"precision format"]
"""

import operator
from logging import getLogger
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....modify.quantizers.functions.add import (
    add_integer,
    construct_essential_config_add_integer,
    get_output_bitwidth_add_integer,
)
from ....modify.quantizers.functions.matmul import (
    bmm_integer,
    construct_essential_config_generic_matmul_integer,
    get_output_bitwidth_bmm_integer,
    matmul_integer,
)
from ....modify.quantizers.functions.relu import (
    construct_essential_config_relu_integer,
    get_output_bitwidth_relu_integer,
    relu_integer,
)
from ....modify.quantizers.layers import (
    AddInteger,
    Conv1dInteger,
    Conv2dInteger,
    LinearInteger,
    ReLUInteger,
)

logger = getLogger(__name__)

QUANTIZED_FUNC_TO_GET_OUTPUT_BITWIDTH_FUNC = {
    add_integer: get_output_bitwidth_add_integer,
    matmul_integer: NotImplementedError(),
    bmm_integer: get_output_bitwidth_bmm_integer,
    relu_integer: get_output_bitwidth_relu_integer,
}


def _set_torch_type_precision_and_format(item: Dict, dtype):
    item["type"] = str(dtype)
    item["precision"] = (torch.finfo(dtype).bits,)
    item["precision_format"] = "(width,)"


def _set_quant_dtype_precision_and_format(item: Dict, config: Dict, config_index: str):
    config_name = config["name"]
    if config_name == "integer":
        item["type"] = "fixed"
        item["precision"] = (
            config[config_index + "_width"],
            config[config_index + "_frac_width"],
        )
        item["precision_format"] = "(width, frac_width)"
    else:
        logger.warning(f"Unrecognized quantization scheme `{config_name}`")


def _set_dtype_before_call_function(node, function, args, kwargs):
    """
    - type
    - precision
    - precision_format
    """
    assert (
        "modify-sw" in node.meta["software"]
    ), "Failed to find 'modify-sw' in metadata['software']. Make sure after run this pass after modifier which record quant_config for each call_function node"
    config = node.meta["software"]["modify-sw"]["config"]
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    if function in (F.relu,):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif function(operator.add, torch.add, torch.matmul, torch.bmm):
        _set_torch_type_precision_and_format(mc_args["data_in_0"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_args["data_in_1"], args[1].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    # ------------------------------------------
    # Quantized format
    # ------------------------------------------
    elif function in (relu_integer,):
        config = construct_essential_config_relu_integer(config)
        output_config = get_output_bitwidth_relu_integer(config)
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], output_config, "data_out"
        )
    elif function in (add_integer,):
        config = construct_essential_config_add_integer(config)
        output_config = get_output_bitwidth_add_integer(config)
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], output_config, "data_out"
        )
    elif function in (bmm_integer,):
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        output_config = get_output_bitwidth_bmm_integer(config=config, x_shape=x_shape)
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], output_config, "data_out"
        )
    elif function in (matmul_integer,):
        # matmul supports broadcasting, but we temporarily treat it as bmm
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        output_config = get_output_bitwidth_bmm_integer(config=config, x_shape=x_shape)
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], output_config, "data_out"
        )
        logger.warning(
            "A quantized `matmul_integer`'s quant_config is constructed as a `bmm_integer`'s quant_config"
        )
    else:
        logger.warning(f"Unrecognized function `{function}` when setting dtype")


def _set_dtype_before_call_module(node, module, args, kwargs):
    """
    - type
    - precision
    - precision_format
    """
    # config = node.meta["software"]["modify-sw"]["config"]
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]

    if type(module) in (nn.ReLU,):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif type(module) in (nn.Linear, nn.Conv1d, nn.Conv2d):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_args["weight"], module.weight.dtype)
        if module.bias is not None:
            _set_torch_type_precision_and_format(mc_args["weight"], module.bias.dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif type(module) in (ReLUInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif type(module) in (AddInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif type(module) in (LinearInteger, Conv1dInteger, Conv2dInteger):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["weight"], config, "weight")
        if module.bias is not None:
            _set_quant_dtype_precision_and_format(mc_args["bias"], config, "bias")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting dtype"
        )


def _set_dtype_before_call_method(node, method_name, args, kwargs):
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    if method_name in ("relu"):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif method_name in ("add", "matmul", "bmm"):
        _set_torch_type_precision_and_format(mc_args["data_in_0"], args[0].dtype)
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], args[1].dtype)
        _set_quant_dtype_precision_and_format(mc_results["data_out"], args[0].dtype)
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting dtype")
