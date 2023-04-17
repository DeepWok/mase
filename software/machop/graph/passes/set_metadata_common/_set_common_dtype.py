"""
hook function for analyzing dtype & quantization info

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["type"&"precision"]
"""
import operator
from copy import deepcopy
from logging import getLogger
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node
from torchvision.ops.stochastic_depth import stochastic_depth

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

TORCH_DTYPE_TO_HW_DTYPE = {
    torch.float32: "float",
    torch.float: "float",
    torch.float64: "float",
    torch.int64: "fixed",
    torch.int: "fixed",
    torch.bool: "fixed",
    torch.Size: "fixed",
}

TORCH_DTYPE_TO_HW_PRECISION = {
    torch.float32: (32,),
    torch.float: (32,),
    torch.float64: (64,),
    torch.int: (32, 0),
    torch.long: (64, 0),
    torch.bool: (1, 0),
    torch.Size: (32, 0),
}

NON_TORCH_DTYPE_TO_HW_DTYPE = {
    float: "float",
    int: "fixed",
    bool: "bool",
    slice: "slice",
}

NON_TORCH_DTYPE_TO_PRECISION = {
    float: (32, 0),
    int: (32, 0),
    bool: (1, 0),
    slice: "NA",
}


def _set_torch_dtype_precision(item: Dict, dtype):
    item["type"] = TORCH_DTYPE_TO_HW_DTYPE[dtype]
    item["precision"] = TORCH_DTYPE_TO_HW_PRECISION[dtype]


def _get_torch_dtype_precision(torch_dtype):
    dtype = TORCH_DTYPE_TO_HW_DTYPE[torch_dtype]
    precision = TORCH_DTYPE_TO_HW_PRECISION[torch_dtype]
    return dtype, precision


def _set_non_torch_dtype_precision(item: Dict, dtype):
    item["type"] = NON_TORCH_DTYPE_TO_HW_DTYPE[dtype]
    item["precision"] = NON_TORCH_DTYPE_TO_PRECISION[dtype]

def _set_non_torch_dtype_precision_and_format(item: Dict, dtype):
    item["type"] = NON_TORCH_DTYPE_TO_HW_DTYPE[dtype]
    item["precision"] = NON_TORCH_DTYPE_TO_PRECISION[dtype]
    item["precision_format"] = NON_TORCH_DTYPE_TO_PRECISION_FORMAT[dtype]

def _get_non_torch_dtype_precision(non_torch_dtype):
    dtype = NON_TORCH_DTYPE_TO_HW_DTYPE[non_torch_dtype]
    precision = NON_TORCH_DTYPE_TO_PRECISION[non_torch_dtype]
    return dtype, precision

def _set_quant_dtype_precision(item: Dict, config: Dict, config_index: str):
    config_name = config["name"]
    if config_name == "integer":
        item["type"] = "fixed"
        item["precision"] = (
            config[config_index + "_width"],
            config[config_index + "_frac_width"],
        )
    else:
        logger.warning(f"Unrecognized quantization scheme `{config_name}`")


def _get_seq_dtype(l: Union[Tuple, List]):
    element_0 = l[0]
    element_0_type = type(element_0)
    if element_0_type == torch.Tensor:
        element_0_torch_dtype = element_0.dtype

    for element in l[1:]:
        assert type(element) == element_0_type
        if element_0_type == torch.Tensor:
            assert element_0_torch_dtype == element.dtype

    if element_0_type == torch.Tensor:
        return element_0_torch_dtype
    else:
        return element_0_type


def _get_dtype_precision(x):
    if isinstance(x, torch.Tensor):
        return _get_torch_dtype_precision(x.dtype)
    elif isinstance(x, tuple(NON_TORCH_DTYPE_TO_HW_DTYPE.keys())):
        return _get_non_torch_dtype_precision(type(x))
    elif isinstance(x, (list, tuple)):
        seq_dtype = _get_seq_dtype(x)
        if seq_dtype in TORCH_DTYPE_TO_HW_DTYPE:
            return (
                TORCH_DTYPE_TO_HW_DTYPE[seq_dtype],
                TORCH_DTYPE_TO_HW_PRECISION[seq_dtype],
            )
        elif seq_dtype in NON_TORCH_DTYPE_TO_HW_DTYPE:
            return (
                NON_TORCH_DTYPE_TO_HW_DTYPE[seq_dtype],
                NON_TORCH_DTYPE_TO_PRECISION[seq_dtype],
            )
    else:
        raise RuntimeError


def _set_type_precision(item: Dict, dtype, precision):
    item["type"] = dtype
    item["precision"] = precision


def _set_dtype_before_call_function(node: Node, function, args, kwargs):
    """
    - type
    - precision
    """
    assert (
        "modify-sw" in node.meta["software"]
    ), "Failed to find 'modify-sw' in metadata['software']. Make sure after run this pass after modifier which record quant_config for each call_function node"
    config = node.meta["software"]["modify-sw"].get("config", None)
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    if function in (
        F.relu,
        F.hardswish,
        F.hardsigmoid,
        F.sigmoid,
        F.silu,
        F.softmax,
        torch.matmul,
        torch.bmm,
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        operator.floordiv,
        torch.floor_divide,
        operator.eq,
        torch.eq,
        torch.mean,
        stochastic_depth,
        torch._assert,
    ):
        if len(node.all_input_nodes) == 1:
            _set_type_precision(mc_args["data_in"], *_get_dtype_precision(args[0]))
        else:
            for i in range(len(node.all_input_nodes)):
                _set_type_precision(
                    mc_args[f"data_in_{i}"], *_get_dtype_precision(args[i])
                )
    elif function in (
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        torch.unbind,
    ):
        if len(node.all_input_nodes) == 1:
            pass
        else:
            for i in range(len(node.all_input_nodes)):
                if isinstance(args[i], torch.Tensor):
                    continue
                else:
                    _set_type_precision(
                        mc_args[f"data_in_{i}"],
                        *_get_dtype_precision(args[i]),
                    )
    elif function in (torch.cat, torch.concat):
        for i in range(len(node.all_input_nodes)):
            if isinstance(args[0][i], torch.Tensor):
                continue
            else:
                _set_type_precision(
                    mc_args[f"data_in_{i}"],
                    *_get_dtype_precision(args[0][i]),
                )
    elif function in (operator.getitem, getattr):
        if len(node.all_input_nodes) == 1:
            pass
        else:
            for i in range(len(node.all_input_nodes)):
                if isinstance(args[i], torch.Tensor):
                    continue
                else:
                    _set_type_precision(
                        mc_args[f"data_in_{i}"],
                        *_get_dtype_precision(args[i]),
                    )

    # ------------------------------------------
    # Quantized format
    # ------------------------------------------
    elif function in (relu_integer,):
        config = construct_essential_config_relu_integer(
            config
        ) | get_output_bitwidth_relu_integer(config)
        _set_quant_dtype_precision(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif function in (add_integer,):
        config = construct_essential_config_add_integer(
            config
        ) | get_output_bitwidth_add_integer(config)
        _set_quant_dtype_precision(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif function in (bmm_integer,):
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        config = config | get_output_bitwidth_bmm_integer(
            config=config, x_shape=x_shape
        )
        _set_quant_dtype_precision(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif function in (matmul_integer,):
        # matmul supports broadcasting, but we temporarily treat it as bmm
        config = construct_essential_config_generic_matmul_integer(config)
        x_shape = args[0].shape
        config = config | get_output_bitwidth_bmm_integer(
            config=config, x_shape=x_shape
        )
        _set_quant_dtype_precision(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
        logger.warning("A quantized `matmul_integer` is treated as a `bmm_integer`")
    # -----------------------------------------
    else:
        logger.warning(f"Unrecognized function `{function}` when setting dtype")


def _set_dtype_after_call_function(node, function, output):
    mc_results = node.meta["common"]["results"]
    if function in (
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        operator.floordiv,
        torch.floor_divide,
        operator.eq,
        torch.eq,
        torch.concat,
        torch.cat,
        torch.unbind,
        torch.mean,
        torch._assert,
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        torch.concat,
        torch.cat,
        torch.unbind,
        operator.getitem,
        stochastic_depth,
        getattr,
    ):
        if output is None:
            pass
        else:
            _set_type_precision(mc_results["data_out"], *_get_dtype_precision(output))
    elif function in (add_integer, matmul_integer, bmm_integer, relu_integer):
        pass
    else:
        logger.warning(f"Unrecognized function `{function}` when setting dtype")


def _set_dtype_before_call_module(node, module, args, kwargs):
    """
    - type
    - precision
    """
    # config = node.meta["software"]["modify-sw"]["config"]
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    module_cls = type(module)
    module_cls = type(module)

    if module_cls in (nn.Embedding,) or isinstance(module_cls, (nn.Embedding,)):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_args["weight"], module.weight.dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (
        nn.ReLU,
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.Sigmoid,
        nn.SiLU,
        nn.GELU,
    ):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (nn.Softmax,):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (nn.Linear, nn.Conv1d, nn.Conv2d):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_args["weight"], module.weight.dtype)
        if module.bias is not None:
            _set_torch_dtype_precision(mc_args["weight"], module.bias.dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (nn.BatchNorm2d,):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_args["weight"], module.weight.dtype)
        _set_torch_dtype_precision(mc_args["bias"], module.bias.dtype)
        _set_torch_dtype_precision(mc_args["running_mean"], module.running_mean.dtype)
        _set_torch_dtype_precision(mc_args["running_var"], module.running_var.dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (nn.LayerNorm,):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_args["weight"], module.weight.dtype)
        _set_torch_dtype_precision(mc_args["bias"], module.bias.dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
    ):
        _set_torch_dtype_precision(mc_args["data_in"], args[0].dtype)
        _set_torch_dtype_precision(mc_results["data_out"], args[0].dtype)
    elif module_cls in (
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    ):
        # logger.debug(
        #     f"module `{module_cls}`'s precision depends on the previous and the next nodes"
        # )
        pass
    elif module_cls in (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.Identity,
    ):
        pass
        # logger.debug(
        #     f"module `{type(module)}`'s precision depends on the previous and the next nodes"
        # )
    elif module_cls in (ReLUInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif module_cls in (AddInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision(mc_args["data_in_0"], config, "data_in")
        _set_quant_dtype_precision(mc_args["data_in_1"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif module_cls in (LinearInteger, Conv1dInteger, Conv2dInteger):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision(mc_args["weight"], config, "weight")
        if module.bias is not None:
            _set_quant_dtype_precision(mc_args["bias"], config, "bias")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif module_cls in (AvgPool2dInteger,):
        config = module.config | module.get_output_bitwidth()
        _set_quant_dtype_precision(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    elif module_cls in (AdaptiveAvgPool2dInteger,):
        config = module.config | module.get_output_bitwidth(x_shape=args[0].shape)
        _set_quant_dtype_precision(mc_args["data_in"], config, "data_in")
        _set_quant_dtype_precision(mc_results["data_out"], config, "data_out")
    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting dtype"
        )


def _set_dtype_before_call_method(node, method_name, args, kwargs):
    mc_args = node.meta["common"]["args"]
    mc_results = node.meta["common"]["results"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "mean",
        "size",
    ):
        if len(node.all_input_nodes) == 1:
            _set_type_precision(mc_args["data_in"], *_get_dtype_precision(args[0]))
        else:
            for i in range(len(node.all_input_nodes)):
                _set_type_precision(
                    mc_args[f"data_in_{i}"], *_get_dtype_precision(args[i])
                )
    elif method_name in (
        "view",
        "reshape",
        "flatten",
        "transpose",
        "permute",
        "unbind",
        "expand",
        "contiguous",
    ):
        if len(node.all_input_nodes) == 1:
            _set_type_precision(mc_args["data_in"], *_get_dtype_precision(args[0]))
        else:
            for i in range(len(node.all_input_nodes)):
                _set_type_precision(
                    mc_args[f"data_in_{i}"], *_get_dtype_precision(args[i])
                )
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting dtype")


def _set_dtype_after_call_method(node, method_name, output):
    mc_results = node.meta["common"]["results"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "size",
    ):
        _set_type_precision(mc_results["data_out"], *_get_dtype_precision(output))
    elif method_name in (
        "view",
        "flatten",
        "permute",
        "transpose",
        "reshape",
        "unbind",
        "contiguous",
        "expand",
    ):
        pass
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting dtype")


# -----------------------------------------
from ._utils import _get_next_available_dtype_info, _get_prev_available_dtype_info


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
            operator.getitem,
            getattr,
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
            nn.Identity,
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
            "unbind",
        ):
            _set_smaller_width_in_neighbors(node, real_target=real_target)
    else:
        pass


def _set_smaller_width_in_neighbors(node, real_target):
    """
    the dtype of current node can be same as the previous and next node,
    so set current node's precision the same as the smaller one
    """
    # fmt: off
    next_available_info = _get_next_available_dtype_info(node=node)
    prev_available_info = _get_prev_available_dtype_info(node=node)
    if next_available_info is not None and prev_available_info is not None:
        if next_available_info["precision"][0] <= prev_available_info["precision"][0]:
            available_info = next_available_info
        else:
            available_info = prev_available_info
    elif next_available_info is not None:
        available_info = next_available_info
    elif prev_available_info is not None:
        available_info = prev_available_info
    else:
        logger.warning(f"Cannot find available dtype & precision info from neighbor nodes for Node {node}")
    # !: This is probably not correct
    for data_in_i, item in node.meta["common"]["args"].items():
        if item["type"] == "NA":
            item["type"]=available_info["type"]
            item["precision"]=available_info["precision"]

    node.meta["common"]["results"]["data_out"]["type"] = available_info["type"]
    node.meta["common"]["results"]["data_out"]["precision"] = available_info["precision"]
    # fmt: on
