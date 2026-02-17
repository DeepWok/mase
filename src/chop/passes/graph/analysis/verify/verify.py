import logging

from chop.passes.graph.utils import vf

from .common_metadata_layers import (
    verify_common_metadata_flatten,
    verify_common_metadata_general,
    verify_common_metadata_input,
    verify_common_metadata_linear,
    verify_common_metadata_output,
    verify_common_metadata_relu,
)

logger = logging.getLogger(__name__)


def verify_node_common_metadata(node):
    """
    Verify the common metadata of a node.

    This function checks the common metadata of a node and performs specific verification based on the mase_op parameter.

    :param node: The node to verify.
    :type node: Node
    :raises ValueError: If the mase_op parameter is unknown.
    """
    verify_common_metadata_general(node.meta["mase"])

    if node.meta["mase"].parameters["common"]["mase_op"] == "placeholder":
        verify_common_metadata_input(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "output":
        verify_common_metadata_output(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "linear":
        verify_common_metadata_linear(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "relu":
        verify_common_metadata_relu(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "flatten":
        verify_common_metadata_flatten(node.meta["mase"])
    elif node.meta["mase"].parameters["common"]["mase_op"] == "constant":
        # Add specific verification for constant operation
        pass
        verify_common_metadata_input(node.meta["mase"])
    else:
        raise ValueError(
            "Unknown mase op: {}".format(
                node.meta["mase"].parameters["common"]["mase_op"]
            )
        )


def verify_common_metadata_analysis_pass(graph, pass_args: dict = {}):
    """Verify pass for mase graph
    This pass is used for verification of MaseGraph.
    It does sanity checks the common metadata of
    each mase node locally and then verify the inter-node
    invariants, particularly for the following:

    :param graph: The input graph to analyze.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the analysis pass (optional).
    :type pass_args: dict

    pass_args is not used in this pass, defaults to {}.

    :return: The analyzed graph and an empty dictionary.
    :rtype: tuple(MaseGraph, Dict)
    """

    # Verify each node in the graph
    for node in graph.fx_graph.nodes:
        verify_node_common_metadata(node)

    # Each node must have a unique name and a unique verilog name
    node_names = []
    node_vf_names = []
    for node in graph.fx_graph.nodes:
        assert node.name not in node_names
        assert vf(node.name) not in node_vf_names
        node_names.append(node.name)
        node_vf_names.append(vf(node.name))

    # Each node must have at most one result
    for node in graph.fx_graph.nodes:
        assert len(node.meta["mase"].parameters["common"]["results"]) <= 1

    # Inter-node verification
    # Each edge between nodes must have the same size
    for node in graph.fx_graph.nodes:
        if len(node.all_input_nodes) > 0:
            for i, args in enumerate(node.args):
                data_in = node.meta["mase"].parameters["common"]["args"][f"data_in_{i}"]
                dst_size = data_in["size"]
                src_size = (
                    data_in["from"]
                    .meta["mase"]
                    .parameters["common"]["results"][f"data_out_0"]["size"]
                )
                assert dst_size == src_size

    return graph, {}


def verify_software_metadata_analysis_pass(graph, pass_args: dict = {}):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks the software metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO

    :param graph: The input graph to analyze.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the analysis pass (optional).
    :type pass_args: dict

    pass_args is not used in this pass, defaults to {}.

    :return: The analyzed graph and an empty dictionary.
    :rtype: tuple(MaseGraph, Dict)

    """
    return graph, {}


def verify_metadata_analysis_pass(graph, pass_args: dict = {}):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks all the metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO

    :param graph: The input graph to analyze.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the analysis pass (optional).
    :type pass_args: dict

    pass_args is not used in this pass, defaults to {}.

    :return: The analyzed graph and an empty dictionary.
    :rtype: tuple(MaseGraph, Dict)
    """
    _, _ = verify_common_metadata_analysis_pass(graph)
    _, _ = verify_software_metadata_analysis_pass(graph)
    return graph, {}
