"""
hook functions for initializing metadata
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


def _empty_common(old_meta):
    meta = {
        "common": {"args": {}, "results": {}},
        "software": old_meta.get("software", {}),
        "hardware": old_meta.get("hardware", {}),
    }
    return meta


def _add_empty_common_args(meta, *arg_names):
    assert len(set(arg_names)) == len(arg_names), f"Duplicated arg_name in {arg_names}"
    for arg_name in arg_names:
        # Here "NA" is used as default since TOML does not support `None`
        meta["common"]["args"][arg_name] = {
            "type": "NA",
            "precision": "NA",
            "size": "NA",
            "from": "NA",
            "is_packed": False,
        }


def _add_empty_common_results(meta, *result_names):
    assert len(set(result_names)) == len(
        result_names
    ), f"Duplicated result_name in {result_names}"
    for result_name in result_names:
        meta["common"]["results"][result_name] = {
            "type": "NA",
            "precision": "NA",
            "size": "NA",
            "is_packed": False,
        }


def _set_empty_metadata_before_call_function(node: Node, function, args, kwargs):
    node.meta = _empty_common(node.meta)
    meta = node.meta
    if function in (
        F.relu,
        relu_integer,
        F.hardswish,
        F.hardsigmoid,
        F.silu,
        F.sigmoid,
        F.gelu,
        F.softmax,
        operator.add,
        torch.add,
        add_integer,
        operator.mul,
        torch.mul,
        operator.eq,
        torch.eq,
        operator.floordiv,
        torch.floor_divide,
        torch.cat,
        torch.concat,
        torch.unbind,
        torch.mean,
        torch.matmul,
        torch.bmm,
        matmul_integer,
        bmm_integer,
        torch.flatten,
        torch.reshape,
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
        OPTDecoder_self_prepare_decoder_attention,
        OPTForCasualLM_compute_loss
        # torch.max,
        # torch.maximum,
        # torch.min,
        # torch.minimum,
    ):
        data_in_names = []
        if len(node.all_input_nodes) == 1:
            data_in_names.append("data_in")
        else:
            for i in range(len(node.all_input_nodes)):
                data_in_names.append(f"data_in_{i}")
        _add_empty_common_args(meta, *data_in_names)
        if function in (torch._assert, _assert_is_none):
            pass
        else:
            _add_empty_common_results(meta, "data_out")
    else:
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
        logger.warning(
            f"Unrecognized function `{function}` when setting empty metadata"
        )


def _set_empty_metadata_before_call_module(node, module, args, kwargs):
    node.meta = _empty_common(node.meta)
    meta = node.meta
    if isinstance(
        module, (nn.ReLU, nn.Hardsigmoid, nn.Hardswish, nn.SiLU, nn.Sigmoid, nn.GELU)
    ):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Softmax,)):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Embedding,)):
        _add_empty_common_args(meta, "data_in", "weight")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Embedding,)):
        _add_empty_common_args(meta, "data_in", "weight")
        _add_empty_common_results(meta, "data_out")
    # elif isinstance(module, (AddInteger,)):
    #     _add_empty_common_args(meta, "data_in_0", "data_in_1")
    #     _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Linear,)):
        _add_empty_common_args(meta, "data_in", "weight", "bias")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Conv1d,)):
        _add_empty_common_args(meta, "data_in", "weight", "bias")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.Conv2d,)):
        _add_empty_common_args(meta, "data_in", "weight", "bias")
        _add_empty_common_results(meta, "data_out")
    # fmt: off
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,)):
        # fmt: on
        _add_empty_common_args(
            meta,
            "data_in",
            "weight",
            "bias",
            "running_mean",
            "running_var",
        )
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (nn.LayerNorm,)):
        _add_empty_common_args(meta, "data_in", "weight", "bias")
        _add_empty_common_results(meta, "data_out")
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
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(
        module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)
    ):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, OPTAttentionInteger):
        _add_empty_common_args(
            meta,
            "data_in_0",  # "hidden_states",
            "data_in_1",  # "attention_mask",
            "weight_k_proj",
            "bias_k_proj",
            "weight_v_proj",
            "bias_v_proj",
            "weight_q_proj",
            "bias_q_proj",
            "weight_out_proj",
            "bias_out_proj",
        )
        _add_empty_common_results(meta, "data_out")  # "output_attention"
    else:
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting empty metadata"
        )


def _set_empty_metadata_before_call_method(node, method_name: str, args, kwargs):
    node.meta = _empty_common(node.meta)
    meta = node.meta
    if method_name in (
        "relu",
        "softmax",
        "add",
        "mul",
        "matmul",
        "bmm",
        "unbind",
        "mean",
        "view",
        "reshape",
        "flatten",
        "transpose",
        "permute",
        "expand",
        "contiguous",
        "size",
    ):
        data_in_names = []
        if len(node.all_input_nodes) == 1:
            data_in_names.append("data_in")
        else:
            for i in range(len(node.all_input_nodes)):
                data_in_names.append(f"data_in_{i}")
        _add_empty_common_args(meta, *data_in_names)
        _add_empty_common_results(meta, "data_out")
    else:
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
        logger.warning(
            f"Unrecognized method name `{method_name}` when setting empty metadata"
        )
