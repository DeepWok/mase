import logging
import math

import toml
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from chop.passes.analysis.utils import (
    is_tensor_constant,
    match_and_filter,
    is_seq_blocks_parameter,
)
from chop.passes.common import (
    MASE_BUILTIN_FUNCS,
    MASE_IMPLICIT_FUNCS,
    MASE_MODULE_RELATED_FUNCS,
)
from chop.passes.metadata.mase_metadata import MaseMetadata
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
            elif isinstance(module, nn.MaxPool1d):
                mase_op = "max_pool1d"
            elif isinstance(module, nn.MaxPool2d):
                mase_op = "max_pool2d"
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


def _update_arg_in_next_node(offset, index, arg_in, next_node, node, keys=None):
    if str(arg_in) == str(node.name):
        assert (
            f"data_in_{index}"
            not in next_node.meta["mase"].parameters["common"]["args"]
        ), f"Adding meta to an existing input: {next_node.name} at input {index}"
        next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"] = dict(
            node.meta["mase"].parameters["common"]["results"]["data_out_0"]
        )
        next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"][
            "from"
        ] = node
        if keys is not None:
            next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"][
                "key"
            ] = keys[index - offset]
    else:
        # If an arg of the next node is a constant, add to the metadata,
        # because this has no edge and cannot be udpated from traversing.
        if (
            "data_in_{index}"
            in next_node.meta["mase"].parameters["common"][
                "args"
                # we also ignore arg_in as a string
            ]
            or isinstance(arg_in, torch.fx.Node)
            or isinstance(arg_in, str)
        ):
            return

        if isinstance(arg_in, float):
            next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"] = {
                "size": [1],
                "type": "float",
                "precision": [32],
                "from": "NA",
                "value": arg_in,
            }
            if keys is not None:
                next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"][
                    "key"
                ] = keys[index - offset]

        elif isinstance(arg_in, int):
            next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"] = {
                "size": [1],
                "type": "fixed",
                "precision": [
                    int(math.ceil(math.log2(max(abs(arg_in), 1)))) + 1,
                    0,
                ],
                "from": "NA",
                "value": arg_in,
            }
            if keys is not None:
                next_node.meta["mase"].parameters["common"]["args"][f"data_in_{index}"][
                    "key"
                ] = keys[index - offset]
        elif isinstance(arg_in, tuple):
            # NOTE: Tuples are generally used to convey shape information
            # (e.g. AdaptiveAvgPool2d's output_size). We don't associate them
            # with metadata as we assume they're pretty much constant and can
            # be considered as a local attribute of the model.
            # Check if all elements of the tuple are fixed constants
            if all(isinstance(x, int) for x in arg_in):
                return
            # TODO: Not really sure how to handle this. It seems to me that slicing a tensor would result to this.
            # Adding a case to skip it first.
            # out_bin[:, :, self.input_mask] evals to getitem(stack, (0, slice(None, None, None), slice(None, None, None)))
            if node.name == "stack":
                logger.warning(
                    "Tuple contains unsupported types! For torch.stack involved"
                )
                return
            logger.warning("Tuple contains unsupported types!")
        elif arg_in is None:
            pass
        elif isinstance(arg_in, dict):
            # TODO: This might overwrite the existing args!! Discuss when observed
            arg_keys = list(arg_in.keys())
            for i, a in enumerate(arg_in.values()):
                _update_arg_in_next_node(index, i, a, next_node, node, keys=arg_keys)
        elif isinstance(arg_in, list):
            # TODO: A risk is that now the input count is a variable but it is fine for now
            # We are recording the element within the list seperately here. (e.g. torch.stack)
            for i, a in enumerate(arg_in):
                if str(a) == str(node):
                    _update_arg_in_next_node(index, i, a, next_node, node)

        else:
            assert False, "Unknown constant arg type."


def analysis_common_parameters(node, dummy_in):
    if node.op == "placeholder":
        node.meta["mase"] = analyse_common_parameters_placeholder(
            node.meta["mase"], dummy_in
        )
    elif node.op == "output":
        node.meta["mase"] = analyse_common_parameters_output(node.meta["mase"])
    elif node.op == "call_module":
        node.meta["mase"] = analyse_common_parameters_module(node.meta["mase"])
    elif node.op == "call_function":
        node.meta["mase"] = analyse_common_parameters_function(node.meta["mase"])
    elif node.op == "call_method":
        node.meta["mase"] = analyse_common_parameters_method(node.meta["mase"])
    elif node.op == "get_attr":
        node.meta["mase"] = analyse_common_parameters_attr(node.meta["mase"])
    else:
        raise ValueError(
            "Unknown mase op: {}".format(
                node.meta["mase"].parameters["common"]["mase_op"]
            )
        )

    if len(node.users) > 0:
        logger.debug(
            "{} : {}".format(
                node.name,
                node.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"],
            )
        )

    # Pass the output shape to the inputs of the next node
    for next_node, _ in node.users.items():
        if "args" not in next_node.meta["mase"].parameters["common"]:
            next_node.meta["mase"].parameters["common"]["args"] = {}

        for index, arg_in in enumerate(next_node.args):
            _update_arg_in_next_node(0, index, arg_in, next_node, node)

        if "stack" in next_node.name:
            if node == next_node.args[0][-1]:
                # We check if the current node is the last node of the list argument of torch.stack
                # We treat all the element within the list seperately here. Hence, offset for keyarg is the len of the list + 1
                offset = len(next_node.args[0]) + 1
                for _index, arg_in in enumerate(next_node.kwargs.values()):
                    index = _index + offset
                    keys = list(next_node.kwargs.keys())
                    _update_arg_in_next_node(
                        offset, index, arg_in, next_node, node, keys=keys
                    )
        else:
            offset = len(next_node.args)
            keys = list(next_node.kwargs.keys())
            for _index, arg_in in enumerate(next_node.kwargs.values()):
                index = _index + offset
                _update_arg_in_next_node(
                    offset, index, arg_in, next_node, node, keys=keys
                )


def graph_iterator_for_metadata(graph, dummy_in=None):
    """
    The size of input and output cannot directly be accessed by functions and some modules.
    This function traverses from the placeholder and passes the metadata through edges.
    """

    # Keep track of which node has been updated
    updated_map = {}

    nodes_in = []
    for node in graph.fx_graph.nodes:
        # if a node has no inputs then it itself represents the input of the graph as a whole
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

            # Either there are no input nodes or all input nodes already processed
            if count == len(node.all_input_nodes):
                analysis_common_parameters(node, dummy_in)
                updated_map[node] = True
                for next_node, x in node.users.items():
                    if next_node not in next_nodes_in:
                        next_nodes_in.append(next_node)

            # Some input nodes not yet processed, process current node in the next iteration
            elif node not in next_nodes_in:
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


def add_common_metadata_analysis_pass(graph, pass_args=None):
    """
    Pass args : initial dummy inputs for inferencing all the shapes for each node
    """
    logger.debug(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    # TODO: FIXEME, this is temporary
    graph = graph_iterator_for_metadata(graph, pass_args)
    return graph
