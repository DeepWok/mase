"""
hook function for recoding tensor shapes during forward propagation

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["size"]
"""
import operator
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.node import Node, map_aggregate
from torchvision.ops.stochastic_depth import stochastic_depth

from ....models.patched_nlp_models.opt_patched.custom_modules import OPTAttentionInteger
from ....models.patched_nlp_models.opt_patched.utils_opt_patched import (
    OPTDecoder_self_prepare_decoder_attention,
    OPTForCasualLM_compute_loss,
)
from ....modify.quantizers.functions import (
    add_integer,
    bmm_integer,
    matmul_integer,
    relu_integer,
)

# from ....modify.quantizers.layers import AddInteger

logger = getLogger(__name__)


def _torch_size_to_tuple(tensor_shape):
    assert isinstance(tensor_shape, torch.Size)
    return tuple(shape_i for shape_i in tensor_shape)


def _get_packed_shape(tensor_list):
    shape = []
    for tensor_i in tensor_list:
        if isinstance(tensor_i, torch.Tensor):
            shape.append(_torch_size_to_tuple(tensor_i.shape))
        elif tensor_i is None:
            shape.append([-1])
        else:
            raise RuntimeError
    return shape


def _get_tuple_shape_and_is_packed(x):
    if isinstance(x, torch.Tensor):
        return _torch_size_to_tuple(x.shape), False
    elif isinstance(x, (int, float)):
        return (1,), False
    elif isinstance(x, torch.Size):
        return (len(x),), False
    elif isinstance(x, (list, tuple)):
        if isinstance(x[0], torch.Tensor):
            return _get_packed_shape(x), True
        else:
            raise RuntimeError
    elif x is None:
        return "NA", False
    else:
        raise RuntimeError


def _set_arg_size_before_call_function(node: Node, function, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.softmax,
        operator.add,
        torch.add,
        add_integer,
        operator.mul,
        torch.mul,
        operator.floordiv,
        torch.floor_divide,
        operator.eq,
        torch.eq,
        torch.unbind,
        torch.mean,
        torch.matmul,
        torch.bmm,
        matmul_integer,
        bmm_integer,
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
        F.dropout3d,
        operator.getitem,
        stochastic_depth,
        torch._assert,
        _assert_is_none,
        getattr,
    ):
        if len(node.all_input_nodes) == 1:
            try:
                (
                    meta_common_args["data_in"]["size"],
                    meta_common_args["data_in"]["is_packed"],
                ) = _get_tuple_shape_and_is_packed(args[0])
                meta_common_args["data_in"].pop("from")
            except RuntimeError():
                raise RuntimeError()
        else:
            for i in range(len(node.all_input_nodes)):
                (
                    meta_common_args[f"data_in_{i}"]["size"],
                    meta_common_args[f"data_in_{i}"]["is_packed"],
                ) = _get_tuple_shape_and_is_packed(args[i])
                meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i].name
    elif function in (torch.cat, torch.concat):
        for i in range(len(node.all_input_nodes)):
            (
                meta_common_args[f"data_in_{i}"]["size"],
                meta_common_args[f"data_in_{i}"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[0][i])
            meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i].name
    elif function in (OPTDecoder_self_prepare_decoder_attention,):
        for i in range(len(node.all_input_nodes)):
            (
                meta_common_args[f"data_in_{i}"]["size"],
                meta_common_args[f"data_in_{i}"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[i])
            meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i].name
    elif function in (OPTForCasualLM_compute_loss,):
        args = list(kwargs.values())
        # args[0]: logits
        # args[1]: labels
        for i in range(len(node.all_input_nodes)):
            (
                meta_common_args[f"data_in_{i}"]["size"],
                meta_common_args[f"data_in_{i}"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[i])
            meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i].name
        pass
    else:
        logger.warning(f"Unrecognized function `{function}` when setting input size")


def _set_result_size_after_call_function(node: Node, function, output):
    meta_common_results = node.meta["common"]["results"]

    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.softmax,
        operator.add,
        torch.add,
        add_integer,
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
        torch.matmul,
        torch.bmm,
        matmul_integer,
        bmm_integer,
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
        F.dropout3d,
        getattr,
        operator.getitem,
        stochastic_depth,
        torch._assert,
        _assert_is_none,
        OPTForCasualLM_compute_loss,
        OPTDecoder_self_prepare_decoder_attention,
    ):
        if output is None:
            return
        else:
            (
                meta_common_results["data_out"]["size"],
                meta_common_results["data_out"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(output)
    else:
        logger.warning(f"Unrecognized function `{function}` when setting output size")


def _set_arg_size_before_call_module(node: Node, module, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    if "data_in" in meta_common_args:
        meta_common_args["data_in"].pop("from")
    if isinstance(
        module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid, nn.GELU)
    ):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)

        # meta_common_args["data_in"].pop("from")
        # meta_common_args["data_in"]["from"] = node.all_input_nodes[0]
    elif isinstance(module, (nn.Softmax,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
    # elif isinstance(module, (AddInteger,)):
    #     meta_common_args["data_in_0"]["size"] = _torch_size_to_tuple(args[0].shape)
    #     meta_common_args["dat_in_0"]["from"] = node.all_input_nodes[0].name
    #     meta_common_args["data_in_1"]["size"] = _torch_size_to_tuple(args[1].shape)
    #     meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1].name
    elif isinstance(module, (nn.Linear,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
        meta_common_args["running_mean"]["size"] = _torch_size_to_tuple(
            module.running_mean.shape
        )
        meta_common_args["running_var"]["size"] = _torch_size_to_tuple(
            module.running_var.shape
        )
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
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
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    elif isinstance(
        module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)
    ):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    elif isinstance(module, OPTAttentionInteger):
        # fmt: off
        meta_common_args["data_in_0"]["size"] = _torch_size_to_tuple(kwargs["hidden_states"].shape)
        meta_common_args["data_in_0"]["from"] = node.all_input_nodes[0].name
        meta_common_args["data_in_1"]["size"] = _torch_size_to_tuple(kwargs["attention_mask"].shape)
        meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1].name
        meta_common_args["weight_k_proj"]["size"] = _torch_size_to_tuple(module.k_proj.weight.shape)
        if module.k_proj.bias is not None:
            meta_common_args["bias_k_proj"]["size"] = _torch_size_to_tuple(module.k_proj.bias.shape)

        meta_common_args["weight_v_proj"]["size"] = _torch_size_to_tuple(module.v_proj.weight.shape)
        if module.v_proj.bias is not None:
            meta_common_args["bias_v_proj"]["size"] = _torch_size_to_tuple(module.v_proj.bias.shape)

        meta_common_args["weight_q_proj"]["size"] = _torch_size_to_tuple(module.q_proj.weight.shape)
        if module.q_proj.bias is not None:
            meta_common_args["bias_q_proj"]["size"] = _torch_size_to_tuple(module.q_proj.bias.shape)

        meta_common_args["weight_out_proj"]["size"] = _torch_size_to_tuple(module.out_proj.weight.shape)
        if module.out_proj.bias is not None:
            meta_common_args["bias_out_proj"]["size"] = _torch_size_to_tuple(module.out_proj.bias.shape)
        # fmt: on

    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_result_size_after_call_module(node: Node, module, output):
    meta_common_results = node.meta["common"]["results"]
    if isinstance(
        module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid, nn.GELU)
    ):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Softmax,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    # elif isinstance(module, (AddInteger,)):
    #     meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Linear,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(
        module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)
    ):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
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
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, OPTAttentionInteger):
        # output = [output_attentions, None]
        meta_common_results["data_out"]["size"] = [
            _torch_size_to_tuple(output[0].shape),
            [1],
        ]
    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_arg_size_before_call_method(node: Node, method_name: str, args, kwargs):
    """
    self_obj.method(self, data_in_1), where 'self' is data_in_0
    """
    meta_common_args = node.meta["common"]["args"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "view",
        "reshape",
        "transpose",
        "permute",
        "flatten",
        "expand",
        "unbind",
        "contiguous",
        "size",
    ):
        if len(node.all_input_nodes) == 1:
            (
                meta_common_args["data_in"]["size"],
                meta_common_args["data_in"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[0])
            meta_common_args["data_in"].pop("from")
        else:
            for i in range(len(node.all_input_nodes)):
                (
                    meta_common_args[f"data_in_{i}"]["size"],
                    meta_common_args[f"data_in_{i}"]["is_packed"],
                ) = _get_tuple_shape_and_is_packed(args[i])
                meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i].name
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")


def _set_result_size_after_call_method(node: None, method_name: str, output):
    meta_common_results = node.meta["common"]["results"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "view",
        "reshape",
        "transpose",
        "permute",
        "flatten",
        "expand",
        "unbind",
        "contiguous",
        "size",
    ):
        (
            meta_common_results["data_out"]["size"],
            meta_common_results["data_out"]["is_packed"],
        ) = _get_tuple_shape_and_is_packed(output)
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")
