import logging

import toml
import torch
import torch.fx as fx
from tabulate import tabulate
from torch import nn

from chop.passes.analysis.utils import (get_input_nodes, get_output_nodes,
                                        match_and_filter)
from chop.passes.common import MASE_BUILTIN_FUNCS, MASE_MODULE_RELATED_FUNCS
from chop.passes.metadata.mase_metadata import MaseMetadata

from .common_metadata_layers import (analyse_common_parameters_constant,
                                     analyse_common_parameters_flatten,
                                     analyse_common_parameters_linear,
                                     analyse_common_parameters_output,
                                     analyse_common_parameters_placeholder,
                                     analyse_common_parameters_relu,
                                     analyse_common_parameters_size,
                                     analyse_common_parameters_view)

logger = logging.getLogger(__name__)


def graph_iterator_for_mase_ops(graph):
    for node in graph.fx_graph.nodes:
        node: fx.Node
        if node.op == "call_module":
            module_name = node.target
            module = graph.modules[module_name]
            node.meta["mase"].parameters["common"]["mase_type"] = "module"
            if isinstance(module, nn.AdaptiveAvgPool1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "adaptiveavgpool1d"
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "adaptiveavgpool2d"
            elif isinstance(module, nn.AdaptiveMaxPool1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "adaptivemaxpool1d"
            elif isinstance(module, nn.AdaptiveMaxPool2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "adaptivemaxpool2d"
            elif isinstance(module, nn.AvgPool1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "avgpool1d"
            elif isinstance(module, nn.AvgPool2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "avgpool2d"
            elif isinstance(module, nn.BatchNorm1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "batchnorm1d"
            elif isinstance(module, nn.BatchNorm2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "batchnorm2d"
            elif isinstance(module, nn.Conv2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "conv2d"
            elif isinstance(module, nn.Conv1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "conv1d"
            elif isinstance(module, nn.LayerNorm):
                node.meta["mase"].parameters["common"]["mase_op"] = "layernorm"
            elif isinstance(module, nn.Linear):
                node.meta["mase"].parameters["common"]["mase_op"] = "linear"
            elif isinstance(module, nn.MaxPool1d):
                node.meta["mase"].parameters["common"]["mase_op"] = "maxpool1d"
            elif isinstance(module, nn.MaxPool2d):
                node.meta["mase"].parameters["common"]["mase_op"] = "maxpool2d"
            elif isinstance(module, nn.ReLU):
                node.meta["mase"].parameters["common"]["mase_op"] = "relu"
            else:
                raise ValueError(f"Unknown node type: {node.target}")

        elif node.op == "call_function":
            # we might have things like mult_1, add_2, so we need to match the pattern
            matching, matched_name = match_and_filter(
                node.name, MASE_BUILTIN_FUNCS + MASE_MODULE_RELATED_FUNCS
            )
            if not matching:
                raise ValueError(f"Unknown call_function node: {node.target}")
            if matched_name in MASE_BUILTIN_FUNCS:
                # if node.target in ["mul", "sub", "add", torch.flatten]:
                node.meta["mase"].parameters["common"]["mase_type"] = "builtin_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            # TODO: we might need to add more functions here
            elif matched_name in MASE_MODULE_RELATED_FUNCS:
                node.meta["mase"].parameters["common"][
                    "mase_type"
                ] = "module_related_func"
                node.meta["mase"].parameters["common"]["mase_op"] = matched_name
            else:
                raise ValueError(f"Unknown node type: {node.target}")

        elif node.op == "call_method":
            if node.name in ["size", "view"]:
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"]["mase_op"] = node.target
            else:
                raise ValueError(f"Unknown node type: {node.name}")

        elif node.op == "placeholder":
            node.meta["mase"].parameters["common"]["mase_type"] = "placeholder"
            node.meta["mase"].parameters["common"]["mase_op"] = "placeholder"

        elif node.op == "get_attr":
            if node.name in ["_tensor_constant0"]:
                node.meta["mase"].parameters["common"]["mase_type"] = "implicit_func"
                node.meta["mase"].parameters["common"]["mase_op"] = "constant"
            else:
                node.meta["mase"].parameters["common"]["mase_type"] = "get_attr"
                raise NotImplementedError(f"Unknown node type: {node.target}")

        elif node.op == "output":
            node.meta["mase"].parameters["common"]["mase_type"] = "output"
            node.meta["mase"].parameters["common"]["mase_op"] = "output"

        else:
            raise ValueError(f"Unknown node type: {node.op}")
    return graph


def analysis_common_parameters(node, dummy_in):
    if node.meta["mase"].parameters["common"]["mase_op"] == "placeholder":
        node.meta["mase"] = analyse_common_parameters_placeholder(
            node.meta["mase"], dummy_in
        )
    elif node.meta["mase"].parameters["common"]["mase_op"] == "output":
        node.meta["mase"] = analyse_common_parameters_output(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "linear":
        node.meta["mase"] = analyse_common_parameters_linear(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "relu":
        node.meta["mase"] = analyse_common_parameters_relu(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "flatten":
        node.meta["mase"] = analyse_common_parameters_flatten(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "view":
        node.meta["mase"] = analyse_common_parameters_view(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "size":
        node.meta["mase"] = analyse_common_parameters_size(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "constant":
        node.meta["mase"] = analyse_common_parameters_constant(node.meta["mase"])
    else:
        raise ValueError(
            "Unknown mase op: {}".format(
                node.meta["mase"].parameters["common"]["mase_op"]
            )
        )

    # Pass the output shape to the inputs of the next node
    for next_node, _ in node.users.items():
        if "args" not in next_node.meta["mase"].parameters["common"]:
            next_node.meta["mase"].parameters["common"]["args"] = {}
        for index, arg_in in enumerate(next_node.args):
            if str(arg_in) == str(node.name):
                assert (
                    f"data_in_{index}"
                    not in next_node.meta["mase"].parameters["common"]["args"]
                ), f"Adding meta to an existing input: {next_node.name} at input {index}"
                next_node.meta["mase"].parameters["common"]["args"][
                    f"data_in_{index}"
                ] = node.meta["mase"].parameters["common"]["results"]["data_out_0"]
                next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"][
                    "from"
                ] = node


def graph_iterator_for_metadata(graph, dummy_in=None):
    """
    The size of input and output cannot directly be accessed by functions and some modules.
    This function traverses from the placeholder and passes the metadata through edges.
    """

    # Keep track of which node has been updated
    updated_map = {}

    nodes_in = []
    for node in graph.fx_graph.nodes:
        if len(node.all_input_nodes) == 0:
            nodes_in.append(node)

    while len(nodes_in) > 0:
        logger.debug(f"Live nodes = {len(nodes_in)}: {nodes_in}")
        next_nodes_in = []
        for node in nodes_in:
            if node in updated_map:
                continue

            count = 0
            for input_node in node.all_input_nodes:
                count += input_node in updated_map
            if count == len(node.all_input_nodes):
                analysis_common_parameters(node, dummy_in)
                updated_map[node] = True
                for next_node, x in node.users.items():
                    if next_node not in next_nodes_in:
                        next_nodes_in.append(next_node)
            elif next_node not in next_nodes_in:
                next_nodes_in.append(node)
        if nodes_in == next_nodes_in:
            raise ValueError("Deadlock detected.")

        nodes_in = next_nodes_in

    assert len(updated_map.keys()) == len(graph.fx_graph.nodes)
    return graph


"""
This is a standard analysis pass that runs at the start of all transform calls

name_style_pass (graph, pass_args)

This follows the the naming convention of
[name]_[style]_pass
add_common_metadata(name)_analysis(style)_pass

passname : {args}

"""


def add_common_metadata_analysis_pass(graph, pass_args=None):
    """
    Pass args : initial dummy inputs for inferencing all the shapes for each node
    """
    graph.nodes_in = get_input_nodes(graph.fx_graph)
    graph.nodes_out = get_output_nodes(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    graph = graph_iterator_for_metadata(graph, pass_args)
    return graph
