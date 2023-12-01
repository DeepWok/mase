import logging

import toml
import torch
import torch.fx as fx
from chop.passes.metadata.mase_metadata import MaseMetadata
from chop.passes.analysis.utils import (
    get_input_nodes,
    get_output_nodes,
)
from torch import nn

from .hardware_metadata_layers import (
    analyse_hardware_parameters_linear,
    analyse_hardware_parameters_relu,
    analyse_hardware_parameters_batch_norm1d,
)

logger = logging.getLogger(__name__)


def analysis_hardware_parameters(node):
    if node.meta["mase"].parameters["hardware"]["is_implicit"]:
        return

    op = node.meta["mase"].parameters["common"]["mase_op"]

    if op == "linear":
        node.meta["mase"] = analyse_hardware_parameters_linear(node.meta["mase"])
    elif op == "relu":
        node.meta["mase"] = analyse_hardware_parameters_relu(node.meta["mase"])
    elif op == "batch_norm1d":
        node.meta["mase"] = analyse_hardware_parameters_batch_norm1d(node.meta["mase"])
    else:
        raise ValueError(f"Unknown mase op: {op}")


"""
This is a standard analysis pass that runs at the start of all transform calls

name_style_pass (graph, pass_args)

This follows the the naming convention of
[name]_[style]_pass
add_hardware_metadata(name)_analysis(style)_pass

passname : {args}

"""


def add_hardware_metadata_analysis_pass(graph, pass_args=None):
    """
    Add hardware metadata for accelerator mapping
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"].parameters["hardware"]["is_implicit"] = False
    graph.nodes_in = get_input_nodes(graph.fx_graph)
    graph.nodes_out = get_output_nodes(graph.fx_graph)

    for node in graph.fx_graph.nodes:
        analysis_hardware_parameters(node)
    return graph
