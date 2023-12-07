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
)
from chop.passes.graph.common import (
    MASE_BUILTIN_FUNCS,
    MASE_IMPLICIT_FUNCS,
    MASE_MODULE_RELATED_FUNCS,
)
from chop.ir.graph.mase_metadata import MaseMetadata
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


def load_arg(a, env):
    return torch.fx.graph.map_arg(a, lambda n: env[n.name])


def fetch_attr(mod, target: str):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def graph_iterator_for_metadata(graph, dummy_in=None, add_value=True):
    """
    largely apated from https://pytorch.org/docs/stable/fx.html
    """

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
    env = {}
    prev_result = None

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


"""
This is a standard analysis pass that runs at the start of all transform calls

name_style_pass (graph, pass_args)

This follows the the naming convention of
[name]_[style]_pass
add_common_metadata(name)_analysis(style)_pass

passname : {args}

"""


# TO DO: placeholder. Need to update iterator for metadata with fx's ShapeProp methodology
def new_graph_iterator_for_metadata(graph, dummy_in=None):
    """
    The size of input and output cannot directly be accessed by functions and some modules.
    This function traverses from the placeholder and passes the metadata through edges.
    """

    print(type(graph.model))
    sp = ShapeProp(graph.model)
    g = sp.propagate()

    return graph


def add_common_metadata_analysis_pass(
    graph, pass_args={"dummy_in": None, "add_value": True}
):
    """add common metadata

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional, "add_value" controls whether tensor values would be added to the meta data, defaults to True
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)
    """
    logger.debug(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    graph = graph_iterator_for_metadata(graph, **pass_args)
    return graph, {}
