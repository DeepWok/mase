"""
hook function for recoding tensor shapes during forward propagation

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["size"]
"""
import operator
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.node import Node, map_aggregate
from torchvision.ops.stochastic_depth import stochastic_depth

from ....modify.quantizers.functions import (
    add_integer,
    bmm_integer,
    matmul_integer,
    relu_integer,
)
from ....modify.quantizers.layers import AddInteger

logger = getLogger(__name__)


def _tuple_shape(tensor_shape):
    return tuple(shape_i for shape_i in tensor_shape)


def _set_arg_size_before_call_function(node: Node, function, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.sigmoid,
    ):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif function in (F.softmax,):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif function in (operator.add, torch.add, add_integer):
        # assert len(node.all_input_nodes) == 2, "Only one input node for add"
        # assert isinstance(args[0], torch.Tensor)
        # assert isinstance(args[1], torch.Tensor)
        if not isinstance(args[0], torch.Tensor):
            meta_common_args["data_in_0"]["size"] = (1,)
            meta_common_args["data_in_0"]["from"] = "const"
        else:
            meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)
            meta_common_args["data_in_0"]["from"] = node.all_input_nodes[0]
        if not isinstance(args[1], torch.Tensor):
            meta_common_args["data_in_1"]["size"] = (1,)
            meta_common_args["data_in_1"]["from"] = "const"
        else:
            meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
            meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1]
    elif function in (operator.mul, torch.mul):
        # assert len(node.all_input_nodes) == 2, "Only one input node for mul"
        # assert isinstance(args[0], torch.Tensor)
        # assert isinstance(args[1], torch.Tensor)
        if not isinstance(args[0], torch.Tensor):
            meta_common_args["data_in_0"]["size"] = (1,)
            meta_common_args["data_in_0"]["from"] = "const"
        else:
            meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)
            meta_common_args["data_in_0"]["from"] = node.all_input_nodes[0]
        if not isinstance(args[1], torch.Tensor):
            meta_common_args["data_in_1"]["size"] = (1,)
            meta_common_args["data_in_1"]["from"] = "const"
        else:
            meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
            meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1]

    elif function in (torch.matmul, torch.bmm, matmul_integer, bmm_integer):
        meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["data_in_0"]["from"] = node.all_input_nodes[0]
        meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
        meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1]
    elif function in (torch.reshape, torch.flatten, torch.transpose, torch.permute):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif function in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    # -----------------------------------------
    elif function in (getattr,):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        if args[1] == "shape":
            meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        else:
            raise RuntimeError
    elif str(function) in ("<built-in function getitem>",):
        if isinstance(args[0], torch.Tensor):
            meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        elif isinstance(args[0], torch.Size):
            meta_common_args["data_in"]["size"] = (len(args[0]),)
        else:
            raise RuntimeError

    # -----------------------------------------
    elif function in (stochastic_depth,):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    else:
        logger.warning(f"Unrecognized function `{function}` when setting size")


def _set_result_size_after_call_function(node: Node, function, output):
    meta_common_results = node.meta["common"]["results"]
    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.sigmoid,
    ):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (F.softmax,):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (operator.add, torch.add, add_integer):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (operator.mul, torch.mul):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (torch.matmul, torch.bmm, matmul_integer, bmm_integer):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (torch.reshape, torch.flatten, torch.transpose, torch.permute):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif function in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    # -----------------------------------------
    elif function in (getattr,):
        if isinstance(output, torch.Tensor):
            meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
        elif isinstance(output, torch.Size):
            meta_common_results["data_out"]["size"] = (len(output),)
        # elif isinstance(output, (list, tuple)):
        #     meta_common_results["data_out"]["size"] = (len(output),)
        else:
            raise RuntimeError()
    elif str(function) in ("<built-in function getitem>",):
        if isinstance(output, torch.Tensor):
            meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
        elif isinstance(output, (int, float)):
            meta_common_results["data_out"]["size"] = (1,)
        else:
            raise RuntimeError()

    # -----------------------------------------
    elif function in (stochastic_depth,):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    else:
        logger.warning(f"Unrecognized function `{function}` when setting size")


def _set_arg_size_before_call_module(node: Node, module, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    if isinstance(module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        # meta_common_args["data_in"].pop("from")
        # meta_common_args["data_in"]["from"] = node.all_input_nodes[0]
    elif isinstance(module, (nn.Softmax,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
    elif isinstance(module, (AddInteger,)):
        meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["dat_in_0"]["from"] = node.all_input_nodes[0]
        meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
        meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1]
    elif isinstance(module, (nn.Linear,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _tuple_shape(module.bias.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _tuple_shape(module.bias.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _tuple_shape(module.bias.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
        meta_common_args["bias"]["size"] = _tuple_shape(module.bias.shape)
        meta_common_args["running_mean"]["size"] = _tuple_shape(
            module.running_mean.shape
        )
        meta_common_args["running_var"]["size"] = _tuple_shape(module.running_var.shape)
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
        meta_common_args["weight"]["size"] = _tuple_shape(module.weight.shape)
        meta_common_args["bias"]["size"] = _tuple_shape(module.bias.shape)
    elif isinstance(
        module,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_result_size_after_call_module(node: Node, module, output):
    meta_common_results = node.meta["common"]["results"]
    if isinstance(module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Softmax,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (AddInteger,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Linear,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif isinstance(
        module,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)

    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_arg_size_before_call_method(node: Node, method_name: str, args, kwargs):
    """
    self_obj.method(self, data_in_1), where 'self' is data_in_0
    """
    meta_common_args = node.meta["common"]["args"]
    if method_name in ("relu", "softmax"):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)  # self
    elif method_name in ("add",):
        meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)  # self
        meta_common_args["data_in_0"]["from"] = "self"
        meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
        meta_common_args["data_in_1"]["from"] = node._input_nodes[0]
    elif method_name in ("matmul", "bmm"):
        meta_common_args["data_in_0"]["size"] = _tuple_shape(args[0].shape)  # self
        meta_common_args["data_in_0"]["from"] = "self"
        meta_common_args["data_in_1"]["size"] = _tuple_shape(args[1].shape)
        meta_common_args["data_in_1"]["from"] = node._input_nodes[0]
    elif method_name in ("view", "reshape", "transpose", "permute", "flatten"):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif method_name in ("contiguous",):
        meta_common_args["data_in"]["size"] = _tuple_shape(args[0].shape)
    elif method_name in ("size"):
        pass
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")


def _set_result_size_after_call_method(node: None, method_name: str, output):
    meta_common_results = node.meta["common"]["results"]
    if method_name in ("relu", "softmax"):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif method_name in ("add",):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif method_name in ("matmul", "bmm"):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif method_name in ("view", "reshape", "transpose", "permute", "flatten"):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif method_name in ("contiguous",):
        meta_common_results["data_out"]["size"] = _tuple_shape(output.shape)
    elif method_name in ("size",):
        pass
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")
