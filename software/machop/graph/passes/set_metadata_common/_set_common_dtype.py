"""
hook function for analyzing dtype & quantization info

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["type"&"precision"&"precision format"]
"""
import operator
from copy import deepcopy
from logging import getLogger
from typing import Callable, Dict, Union

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
    AdaptiveAvgPool2dInteger,
    AddInteger,
    AvgPool2dInteger,
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

INDEX_TO_POSSIBLE_ARG_NAMES = {0: ("data_in", "data_in_0"), 1: ("weight", "data_in_1")}


def _set_torch_type_precision_and_format(item: Dict, dtype):
    item["type"] = str(dtype)
    item["precision"] = (torch.finfo(dtype).bits,)
    item["precision_format"] = "(width,)"


def _set_non_torch_type_precision_and_format(item: Dict, dtype):
    item["type"] = str(dtype)
    item["precision"] = (32,)  # 32-bit floating-point by default
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
    config = node.meta["software"]["modify-sw"].get("config", None)
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    if function in (F.relu,):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif function in (operator.add, torch.add, torch.matmul, torch.bmm):
        _set_torch_type_precision_and_format(mc_args["data_in_0"], args[0].dtype)
        if not isinstance(args[1], torch.Tensor):
            _set_non_torch_type_precision_and_format(
                mc_args["data_in_1"], type(args[1])
            )
        else:
            _set_torch_type_precision_and_format(mc_args["data_in_1"], args[1].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif function in (
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
    ):
        logger.debug(
            f"function `{function}`'s precision depends on the previous and the next nodes"
        )
    # ------------------------------------------
    # Quantized format
    # ------------------------------------------
    elif function in (relu_integer,):
        config = construct_essential_config_relu_integer(
            config
        ) | get_output_bitwidth_relu_integer(config)
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif function in (add_integer,):
        config = construct_essential_config_add_integer(
            config
        ) | get_output_bitwidth_add_integer(config)
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif function in (bmm_integer,):
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        config = config | get_output_bitwidth_bmm_integer(
            config=config, x_shape=x_shape
        )
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif function in (matmul_integer,):
        # matmul supports broadcasting, but we temporarily treat it as bmm
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        config = config | get_output_bitwidth_bmm_integer(
            config=config, x_shape=x_shape
        )
        _set_quant_dtype_precision_and_format(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision_and_format(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
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
    elif type(module) in (nn.BatchNorm2d,):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_args["weight"], module.weight.dtype)
        _set_torch_type_precision_and_format(mc_args["bias"], module.bias.dtype)
        _set_torch_type_precision_and_format(
            mc_args["running_mean"], module.running_mean.dtype
        )
        _set_torch_type_precision_and_format(
            mc_args["running_var"], module.running_var.dtype
        )
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif type(module) in (nn.LayerNorm,):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_args["weight"], module.weight.dtype)
        _set_torch_type_precision_and_format(mc_args["bias"], module.bias.dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif type(module) in (
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
    ):
        _set_torch_type_precision_and_format(mc_args["data_in"], args[0].dtype)
        _set_torch_type_precision_and_format(mc_results["data_out"], args[0].dtype)
    elif type(module) in (
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    ):
        logger.debug(
            f"module `{type(module)}`'s precision depends on the previous and the next nodes"
        )
    elif type(module) in (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d):
        logger.debug(
            f"module `{type(module)}`'s precision depends on the previous and the next nodes"
        )
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
    elif type(module) in (AvgPool2dInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision_and_format(
            mc_results["data_out"], config, "data_out"
        )
    elif type(module) in (AdaptiveAvgPool2dInteger,):
        config = module.config | module.get_output_bitwidth(x_shape=args[0].shape)
        _set_quant_dtype_precision_and_format(mc_args["data_in"], config, "data_in")
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
    elif method_name in ("view", "reshape", "flatten", "transpose", "permute"):
        logger.debug(
            f"Method {method_name}'s precision depends on the previous and the next nodes"
        )
    elif method_name in ("contiguous",):
        logger.debug(
            f"Method {method_name}'s precision depends on the previous and the next nodes"
        )
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting dtype")


# -----------------------------------------


def _set_dtype_of_nodes_depending_on_neighbors(
    node, real_target: Union[nn.Module, Callable, str]
):
    if node.op == "call_function":
        if real_target in (
            torch.reshape,
            torch.flatten,
            torch.permute,
            torch.transpose,
            F.dropout,
            F.dropout1d,
            F.dropout2d,
            F.dropout3d,
        ):
            _set_smaller_width_in_neighbors(node, real_target=real_target)
    elif node.op == "call_module":
        real_target_cls = type(real_target)
        if real_target_cls in (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
        ):
            _set_smaller_width_in_neighbors(node, real_target=real_target)
    elif node.op == "call_method":
        if real_target in (
            "view",
            "reshape",
            "flatten",
            "transpose",
            "permute",
            "contiguous",
        ):
            # _set_smaller_width_in_neighbors(node, real_target=real_target)
            pass
    else:
        pass


def _set_smaller_width_in_neighbors(node, real_target):
    """
    the dtype of current node can be same as the previous and next node,
    so set current node's precision the same as the smaller one
    """
    if type(real_target) in (
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
    ) or real_target in (
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
        F.dropout3d,
        "view",
        "flatten",
        "reshape",
        "transpose",
        "permute",
        "contiguous",
    ):
        # these OPs have only one input node
        prev_node = node.all_input_nodes[0]
        if prev_node.op in ("call_function", "call_module", "call_method"):
            prev_node_data_out_cm = prev_node.meta["common"]["results"]["data_out"]
            prev_node_data_out_precision = prev_node_data_out_cm.get("precision", "NA")
        else:
            prev_node_data_out_precision = "NA"

        next_node = node.next
        if next_node.op in ("call_function", "call_module", "call_method"):
            next_node_args_cm = next_node.meta["common"]["args"]

            if node in next_node.all_input_nodes:
                index = next_node.all_input_nodes.index(node)
                arg_name = None
                for arg_name_i in INDEX_TO_POSSIBLE_ARG_NAMES[index]:
                    if arg_name_i in next_node_args_cm:
                        arg_name = arg_name_i
                        break
                next_node_data_in_cm = next_node_args_cm[arg_name]
                next_node_data_in_precision = next_node_data_in_cm.get(
                    "precision", "NA"
                )
            else:
                next_node_data_in_precision = "NA"
        else:
            next_node_data_in_precision = "NA"

        if prev_node_data_out_precision != "NA" or next_node_data_in_precision != "NA":
            if (
                prev_node_data_out_precision != "NA"
                and next_node_data_in_precision != "NA"
            ):
                if next_node_data_in_precision[0] < prev_node_data_out_precision[0]:
                    smaller_node_data_in = next_node_data_in_cm
                else:
                    smaller_node_data_in = prev_node_data_out_cm
            elif next_node_data_in_precision != "NA":
                smaller_node_data_in = next_node_data_in_cm
            else:
                # prev_node_data_out_width != "NA":
                smaller_node_data_in = prev_node_data_out_cm
            # fmt: off
            node.meta["common"]["args"]["data_in"]["type"] = smaller_node_data_in["type"]
            node.meta["common"]["args"]["data_in"]["precision"] = smaller_node_data_in["precision"]
            node.meta["common"]["args"]["data_in"]["precision_format"] = smaller_node_data_in["precision_format"]
            node.meta["common"]["results"]["data_out"] = deepcopy(node.meta["common"]["args"]["data_in"])
            # fmt: on
        else:
            logger.error(
                f"Both the prev and next nodes' precision of Node {node} ({real_target}) are 'NA'"
            )
    else:
        logger.warning(f"Node {node}'s dtype & precision is not set.")
