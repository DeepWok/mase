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
from torch.fx.node import Node, map_aggregate

from ....modify.quantizers.functions import (
    add_integer,
    bmm_integer,
    matmul_integer,
    relu_integer,
)
from ....modify.quantizers.layers import AddInteger

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
            "precision_format": "NA",
            "size": "NA",
            "from": "NA",
        }


def _add_empty_common_results(meta, *result_names):
    assert len(set(result_names)) == len(
        result_names
    ), f"Duplicated result_name in {result_names}"
    for result_name in result_names:
        meta["common"]["results"][result_name] = {
            "type": "NA",
            "precision": "NA",
            "precision_format": "NA",
            "size": "NA",
        }


def _set_empty_metadata_before_call_function(node, function, args, kwargs):
    node.meta = _empty_common(node.meta)
    meta = node.meta
    if function in (F.relu, relu_integer):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif function in (operator.add, torch.add, add_integer):
        _add_empty_common_args(meta, "data_in_0", "data_in_1")
        _add_empty_common_results(meta, "data_out")
    elif function in (torch.matmul, torch.bmm, matmul_integer, bmm_integer):
        _add_empty_common_args(meta, "data_in_0", "data_in_1")
        _add_empty_common_results(meta, "data_out")
    elif function in (torch.flatten, torch.reshape, torch.transpose, torch.permute):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif function in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d):
        _add_empty_common_args(meta, "data_in")
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
    if isinstance(module, (nn.ReLU)):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif isinstance(module, (AddInteger,)):
        _add_empty_common_args(meta, "data_in_0", "data_in_1")
        _add_empty_common_results(meta, "data_out")
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
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    else:
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting empty metadata"
        )


def _set_empty_metadata_before_call_method(node, method_name: str, args, kwargs):
    node.meta = _empty_common(node.meta)
    meta = node.meta
    if method_name in ("relu",):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif method_name in ("add"):
        _add_empty_common_args(meta, "data_in_0", "data_in_1")
        _add_empty_common_results(meta, "data_out")
    elif method_name in ("matmul", "bmm"):
        _add_empty_common_args(meta, "data_in_0", "data_in_1")
        _add_empty_common_results(meta, "data_out")
    elif method_name in ("view", "reshape", "flatten", "transpose", "permute"):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif method_name in ("contiguous",):
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
    elif method_name in ("size",):
        pass
    else:
        _add_empty_common_args(meta, "data_in")
        _add_empty_common_results(meta, "data_out")
        logger.warning(
            f"Unrecognized method name `{method_name}` when setting empty metadata"
        )
