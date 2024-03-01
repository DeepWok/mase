import os
from datetime import datetime as dt
from glob import glob

from copy import copy, deepcopy
import logging

import cv2
import numpy as np
import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.autograd import Variable            

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from .utils import create_new_module

QUANTIZEABLE_OP = (
    # "add",
    # "bmm",
    # "conv1d",
    "conv2d",
    # "matmul",
    # "mul",
    "linear",
    # "relu",
    # "sub",
)

logger = logging.getLogger(__name__)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def graph_fake_quantize_by_type(graph, config: dict):
    """
    This function applies fake quantization to the graph by type.
    """

    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        # node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            # update_quant_meta_param(node, node_config, get_mase_op(node))
        # elif get_mase_type(node) in [
        #     "builtin_func",
        #     "module_related_func",
        # ]:
        #     new_f, args, kwargs = create_new_fn(node, node_config)
        #     with graph.fx_graph.inserting_before(node):
        #         new_node = graph.fx_graph.call_function(new_f, args, kwargs)
        #         new_node.name = node.name
        #         new_node.meta["mase"] = copy(node.meta["mase"])
        #         # new_node.meta["mase"].node -> new_node
        #         relink_node_meta(new_node, model=graph.model)
        #         update_quant_meta_param(new_node, node_config, get_mase_op(node))
        #         node.replace_all_uses_with(new_node)
        #     graph.fx_graph.erase_node(node)
    return graph


def graph_fake_quantize_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        # print(node_config)
        if node_config["name"] is None:
            continue
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
        # elif get_mase_type(node) in [
        #     "builtin_func",
        #     "module_related_func",
        # ]:
        #     new_f, args, kwargs = create_new_fn(node, node_config)
        #     with graph.fx_graph.inserting_before(node):
        #         new_node = graph.fx_graph.call_function(new_f, args, kwargs)
        #         new_node.name = node.name
        #         new_node.meta["mase"] = copy(node.meta["mase"])
        #         relink_node_meta(new_node, model=graph.model)
        #         update_quant_meta_param(new_node, node_config, get_mase_op(node))
        #         node.replace_all_uses_with(new_node)
        #     graph.fx_graph.erase_node(node)
        #     logger.debug(
        #         f"Quantized function: {node.target} with config: {node_config}"
        #     )
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    return graph


def fake_quantize_transform_pass(graph, pass_args=None):
    """
    This function applies the fake quantization transform pass to the graph.
    """

    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_fake_quantize_by_type(graph, pass_args)
        case "name":
            graph = graph_fake_quantize_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph
    