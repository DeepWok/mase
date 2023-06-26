import logging
import toml
import torch

from torch import nn
from tabulate import tabulate


from chop.passes.metadata.mase_metadata import MaseMetadata
from chop.passes.analysis.utils import (
    get_input_nodes,
    get_output_nodes,
    match_and_filter,
)
from .common_metadata_layers import (
    analyse_common_parameters_placeholder,
    analyse_common_parameters_output,
    analyse_common_parameters_linear,
    analyse_common_parameters_flatten,
    analyse_common_parameters_relu,
)
from chop.passes.common import MASE_BUILDIN_FUNCS, MASE_MODULE_RELATED_FUNCS

logger = logging.getLogger(__name__)


def graph_iterator_for_mase_ops(graph):
    for node in graph.fx_graph.nodes:
        if node.op == "call_module":
            module_name = node.target
            module = graph.modules[module_name]
            node.mase_type = "module"
            if isinstance(module, nn.Linear):
                node.mase_op = "linear"
            elif isinstance(module, nn.Conv2d):
                node.mase_op = "conv2d"
            elif isinstance(module, nn.Conv1d):
                node.mase_op = "conv1d"
            elif isinstance(module, nn.ReLU):
                node.mase_op = "relu"
            else:
                raise ValueError(f"Unknown node type: {node.target}")

        elif node.op == "call_function":
            # we might have things like mult_1, add_2, so we need to match the pattern
            matching, matched_name = match_and_filter(
                node.name, MASE_BUILDIN_FUNCS + MASE_MODULE_RELATED_FUNCS
            )
            if not matching:
                raise ValueError(f"Unknown call_function node: {node.target}")
            if matched_name in MASE_BUILDIN_FUNCS:
                # if node.target in ["mul", "sub", "add", torch.flatten]:
                node.mase_type = "builtin_funcs"
                node.mase_op = matched_name
            # TODO: we might need to add more functions here
            elif matched_name in MASE_MODULE_RELATED_FUNCS:
                node.mase_type = "module_related_funcs"
                node.mase_op = matched_name
            else:
                raise ValueError(f"Unknown node type: {node.target}")

        elif node.op == "call_method":
            if node.name in ["size", "view"]:
                node.mase_type = "implicit_funcs"
                node.mase_op = node.target
            else:
                raise ValueError(f"Unknown node type: {node.name}")

        elif node.op == "placeholder":
            node.mase_type = "placeholder"
            node.mase_op = "placeholder"

        elif node.op == "get_attr":
            node.mase_type = "get_attr"
            raise NotImplementedError(f"Unknown node type: {node.target}")

        elif node.op == "output":
            node.mase_type = "output"
            node.mase_op = "output"

        else:
            raise ValueError(f"Unknown node type: {node.op}")
    return graph


def graph_iterator_inspect_node(graph):
    headers = ["Node name", "Fx Node op", "Mase type", "Mase op"]
    rows = []
    for node in graph.fx_graph.nodes:
        rows.append([node.name, node.op, node.mase_type, node.mase_op])
    logger.debug("Inspecting graph [add_common_metadata_analysis_pass]")
    logger.debug(tabulate(rows, headers=headers))
    return graph


def analysis_common_parameters(node, dummy_in):
    if node.mase_op == "placeholder":
        node.meta = analyse_common_parameters_placeholder(node.meta, dummy_in)
    elif node.mase_op == "output":
        node.meta = analyse_common_parameters_output(node.meta)
    elif node.mase_op == "linear":
        node.meta = analyse_common_parameters_linear(node.meta)
    elif node.mase_op == "relu":
        node.meta = analyse_common_parameters_relu(node.meta)
    elif node.mase_op == "flatten":
        node.meta = analyse_common_parameters_flatten(node.meta)
    else:
        raise ValueError(f"Unknown mase op: {node.mase_op}")

    # Pass the output shape to the inputs of the next node
    for next_node, _ in node.users.items():
        if "args" not in next_node.meta.parameters["common"]:
            next_node.meta.parameters["common"]["args"] = {}
        for index, arg_in in enumerate(next_node.args):
            if str(arg_in) == str(node.name):
                assert (
                    f"data_in_{index}"
                    not in next_node.meta.parameters["common"]["args"]
                ), f"Adding meta to an existing input: {next_node.name} at input {index}"
                next_node.meta.parameters["common"]["args"][
                    f"data_in_{index}"
                ] = node.meta.parameters["common"]["results"]["data_out_0"]
                next_node.meta.parameters["common"]["args"][f"data_in_{index}"][
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
                count = input_node in updated_map
            if count == len(node.all_input_nodes):
                analysis_common_parameters(node, dummy_in)
                updated_map[node] = True
                for next_node, x in node.users.items():
                    next_nodes_in.append(next_node)
            else:
                next_nodes_in.append(node)
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
    for node in graph.fx_graph.nodes:
        node.meta = MaseMetadata(
            node=node,
            model=graph.model,
            fx_graph=graph.fx_graph,
        )
    # This has to be before init parameters
    graph.nodes_in = get_input_nodes(graph.fx_graph)
    graph.nodes_out = get_output_nodes(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    graph = graph_iterator_for_metadata(graph, pass_args)
    graph = graph_iterator_inspect_node(graph)
    return graph
