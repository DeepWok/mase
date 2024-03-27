import logging
import math

import toml
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from chop.passes.graph.analysis.utils import (
    is_tensor_constant,
    match_and_filter,
    is_seq_blocks_parameter,
    get_input_nodes,
    get_output_nodes,
)
from chop.passes.graph.common import (
    MASE_BUILTIN_FUNCS,
    MASE_IMPLICIT_FUNCS,
    MASE_MODULE_RELATED_FUNCS,
)
from chop.ir.graph.mase_metadata import MaseMetadata
from chop.passes.graph.analysis.utils import fetch_attr, load_arg
from tabulate import tabulate
from torch import nn

from .common_metadata_layers import (
    analyse_common_parameters_attr,
    analyse_common_parameters_function,
    analyse_common_parameters_method,
    analyse_common_parameters_module,
    analyse_common_parameters_output,
    analyse_common_parameters_placeholder,
)

logger = logging.getLogger(__name__)


def graph_iterator_for_mase_ops(graph):
    for node in graph.fx_graph.nodes:
        node: fx.Node
        if node.op == "call_module":
            module_name = node.target
            module = graph.modules[module_name]
            mase_type = "module_related_func"
            if isinstance(module, nn.AdaptiveAvgPool1d):
                mase_op = "adaptive_avg_pool1d"
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                mase_op = "adaptive_avg_pool2d"
            elif isinstance(module, nn.AdaptiveMaxPool1d):
                mase_op = "adaptive_max_pool1d"
            elif isinstance(module, nn.AdaptiveMaxPool2d):
                mase_op = "adaptive_max_pool2d"
            elif isinstance(module, nn.AvgPool1d):
                mase_op = "avg_pool1d"
            elif isinstance(module, nn.AvgPool2d):
                mase_op = "avg_pool2d"
            elif isinstance(module, nn.MaxPool1d):
                mase_op = "max_pool1d"
            elif isinstance(module, nn.MaxPool2d):
                mase_op = "max_pool2d"
            elif isinstance(module, nn.BatchNorm1d):
                mase_type = "module"
                mase_op = "batch_norm1d"
            elif isinstance(module, nn.BatchNorm2d):
                mase_type = "module"
                mase_op = "batch_norm2d"
            elif isinstance(module, nn.Conv2d):
                mase_op = "conv2d"
            elif isinstance(module, nn.Conv1d):
                mase_op = "conv1d"
            elif isinstance(module, nn.LayerNorm):
                mase_op = "layer_norm"
            elif isinstance(module, nn.Linear):
                mase_op = "linear"
            elif isinstance(module, nn.ReLU):
                mase_op = "relu"
            elif isinstance(module, nn.Hardtanh):  # TODO: This is not implemented yet
                mase_op = "hardtanh"
            elif isinstance(module, nn.Embedding):
                mase_type = "implicit_func"
                mase_op = "embedding"
            elif isinstance(module, tuple(graph.model.patched_custom_layers)):
                mase_op = "patched_custom_layers"
            # NOTE: The ones below were added to support MobileNetV2 and MobileNetV3.
            # These don't show up when printing the fx.graph.
            elif isinstance(module, nn.ReLU6):
                mase_op = "relu6"
            elif isinstance(module, nn.Dropout):
                mase_op = "dropout"
            elif isinstance(module, nn.Hardswish):
                mase_op = "hardswish"
            elif isinstance(module, nn.Hardsigmoid):
                mase_op = "hardsigmoid"
            # TODO: temporary. Support all patched attention layers
            elif "attention" in module.__name__.lower():
                mase_op = "attention"
            else:
                raise ValueError(f"Unknown node type: {node.target}")
            node.meta["mase"].parameters["common"]["mase_type"] = mase_type
            node.meta["mase"].parameters["common"]["mase_op"] = mase_op

        elif node.op == "call_function":
            # we might have things like mult_1, add_2, so we need to match the pattern
            matching, matched_name = match_and_filter(
                node.name,
                MASE_BUILTIN_FUNCS
                + MASE_MODULE_RELATED_FUNCS
                + MASE_IMPLICIT_FUNCS
                + graph.model.patched_op_names,
            )
            if not matching:
                raise ValueError(
                    f"Unknown call_function node: {node.target} with name {node.name}"
                )
            if matched_name in MASE_BUILTIN_FUNCS:
                node.meta["mase"].parameters["common"]["mase_type"] = "builtin_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            # TODO: we might need to add more functions here
            elif matched_name in MASE_MODULE_RELATED_FUNCS:
                node.meta["mase"].parameters["common"][
                    "mase_type"
                ] = "module_related_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            elif matched_name in MASE_IMPLICIT_FUNCS:
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            elif matched_name in graph.model.patched_op_names:
                node.meta["mase"].parameters["common"]["mase_type"] = "patched_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            else:
                raise ValueError(f"Unknown node type: {node.target}")

        elif node.op == "call_method":
            # we might have things like size_1, size_2, so we need to match the pattern
            # TODO: might need to add this for others as well.
            matching, matched_name = match_and_filter(node.name, MASE_IMPLICIT_FUNCS)
            if not matching:
                raise ValueError(f"Unknown node type: {node.name}")
            if matched_name in MASE_IMPLICIT_FUNCS:
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"]["mase_op"] = node.target

        elif node.op == "placeholder":
            node.meta["mase"].parameters["common"]["mase_type"] = "placeholder"
            node.meta["mase"].parameters["common"]["mase_op"] = "placeholder"

        elif node.op == "get_attr":
            if node.name in ["_tensor_constant0"] or is_tensor_constant(node.name):
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"]["mase_op"] = "constant"
            elif is_seq_blocks_parameter(node.name):
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"][
                    "mase_op"
                ] = "constant"  # TODO: ??? what to assign here
            else:
                node.meta["mase"].parameters["common"]["mase_type"] = "get_attr"
                # raise NotImplementedError(f"Unknown node type: {node.target}")

        elif node.op == "output":
            node.meta["mase"].parameters["common"]["mase_type"] = "output"
            node.meta["mase"].parameters["common"]["mase_op"] = "output"

        else:
            raise ValueError(f"Unknown node type: {node.op}")
    return graph


def graph_iterator_for_metadata(
    graph, dummy_in=None, add_value=True, force_device_meta=False
):
    """
    largely apated from https://pytorch.org/docs/stable/fx.html
    """

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
    env = {}
    prev_result = None

    # force everything to be on device="meta"
    if force_device_meta:
        dummy_in = {k: v.to("meta") for k, v in dummy_in.items()}
        model = model.to("meta")

    for node in graph.fx_graph.nodes:
        args, kwargs = None, None
        if node.op == "placeholder":
            result = dummy_in[node.name]
            analyse_fn = analyse_common_parameters_placeholder
        elif node.op == "get_attr":
            result = fetch_attr(model, node.target)
            analyse_fn = analyse_common_parameters_attr
        elif node.op == "call_function":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = node.target(*args, **kwargs)
            analyse_fn = analyse_common_parameters_function
        elif node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = getattr(self_obj, node.target)(*args, **kwargs)
            analyse_fn = analyse_common_parameters_method
        elif node.op == "call_module":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = modules[node.target](*args, **kwargs)
            analyse_fn = analyse_common_parameters_module
        elif node.op == "output":
            analyse_fn = analyse_common_parameters_output

        # This is the only code specific to shape propagation.
        # you can delete this `if` branch and this becomes
        # a generic GraphModule interpreter.
        # if isinstance(result, torch.Tensor):
        #     node.shape = result.shape
        #     node.dtype = result.dtype

        node.meta["mase"] = analyse_fn(
            node.meta["mase"], result, args, kwargs, add_value=add_value
        )
        env[node.name] = result

    return graph


def _add_graph_metadata(graph):
    """
    Register graph-level metadata
    """
    graph.meta["mase"]["common"] = {
        "nodes_in": [],
        "nodes_out": [],
        "args": [],
        "results": [],
    }
    graph.meta["mase"]["common"]["nodes_in"] = get_input_nodes(graph.fx_graph)
    graph.meta["mase"]["common"]["nodes_out"] = get_output_nodes(graph.fx_graph)

    graph.meta["mase"]["common"]["args"] = {}
    for node in graph.meta["mase"]["common"]["nodes_in"]:
        for arg, arg_info in node.meta["mase"]["common"]["args"].items():
            if "data" in arg:
                graph.meta["mase"]["common"]["args"][arg] = arg_info

    graph.meta["mase"]["common"]["results"] = {}
    for node in graph.meta["mase"]["common"]["nodes_out"]:
        for result, result_info in node.meta["mase"]["common"]["results"].items():
            if "data" in result:
                graph.meta["mase"]["common"]["results"][result] = result_info

    return graph


def add_common_metadata_analysis_pass(
    graph, pass_args={"dummy_in": None, "add_value": True, "force_device_meta": False}
):
    """add common metadata

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional, "add_value" controls whether tensor values would be added to the meta data, defaults to True
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    The common metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.

    - common
        - mase_op -> str : the mase op of the node, e.g. placeholder, linear, relu
        - mase_type -> str : the mase type of the node, e.g. module, builtin_func, module_related_func
        - args -> {}
             - $name : name of the arg
               (if the arg is a tensor)
                 - type -> type of the arg, e.g. fixed point or float
                 - precision -> format of the type, e.g. (10, 5)
                 - shape -> shape of the arg
               (if the arg is not a tensor)
                 - value of the arg
        - results -> {}
             - $name : name of the result
               (if the result is a tensor)
                 - type -> type of the result, e.g. fixed point or float
                 - precision -> format of the type, e.g. (10, 5)
                 - shape -> shape of the result
               (if the result is not a tensor)
                 - value of the result

    Examples:

    A linear layer in a mase graph:

    .. code-block:: shell

        %fc1 : [num_users=1] = call_module[target=fc1](args = (%flatten,), kwargs = {})


    A linear layer after this pass:

    .. code-block:: JSON

        {
            "common": {
                "mase_type": "module_related_func",
                "mase_op": "linear",
                "args": {
                    "data_in_0": {
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                        "type": "float",
                        "precision": [32],
                    },
                    "weight": {"type": "float", "precision": [32], "shape": [784, 784]},
                    "bias": {"type": "float", "precision": [32], "shape": [784]},
                },
                "results": {
                    "data_out_0": {
                        "type": "float",
                        "precision": [32],
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                    }
                },
            },
            "software": {},
            "hardware": {},
        }


    A relu layer in a mase graph:

    .. code-block:: shell

        %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%fc1,), kwargs = {inplace: False})


    A relu layer after this pass:

    .. code-block:: JSON

        {
            "common": {
                "mase_type": "module_related_func",
                "mase_op": "relu",
                "results": {
                    "data_out_0": {
                        "type": "float",
                        "precision": [32],
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                    }
                },
                "args": {
                    "data_in_0": {
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                        "type": "float",
                        "precision": [32],
                    },
                    "inplace": False,
                },
            },
            "software": {},
            "hardware": {},
        }


    A flatten op in a mase graph:

    .. code-block:: shell

        %flatten : [num_users=1] = call_function[target=torch.flatten](args = (%x,), kwargs = {start_dim: 1, end_dim: -1})


    A flatten op after this pass:


    .. code-block:: JSON

        {
            "common": {
                "mase_type": "implicit_func",
                "mase_op": "flatten",
                "results": {
                    "data_out_0": {
                        "type": "float",
                        "precision": [32],
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                    }
                },
                "args": {
                    "data_in_0": {
                        "shape": [1, 28, 28],
                        "torch_dtype": torch.float32,
                        "type": "float",
                        "precision": [32],
                    },
                    "start_dim": 1,
                    "end_dim": -1,
                },
            },
            "software": {},
            "hardware": {},
        }

    """

    logger.debug(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    graph = graph_iterator_for_metadata(graph, **pass_args)
    graph = _add_graph_metadata(graph)
    return graph, {}
