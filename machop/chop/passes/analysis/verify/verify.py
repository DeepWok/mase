import logging
from chop.passes.utils import vf


from .common_metadata_layers import (
    verify_common_metadata_general,
    verify_common_metadata_linear,
    verify_common_metadata_relu,
    verify_common_metadata_placeholder,
    verify_common_metadata_output,
    verify_common_metadata_flatten,
)
from .hardware_metadata_layers import (
    verify_hardware_metadata_general,
    verify_hardware_metadata_linear,
    verify_hardware_metadata_relu,
    # verify_hardware_metadata_placeholder,
    # verify_hardware_metadata_output,
    # verify_hardware_metadata_flatten,
)


logger = logging.getLogger(__name__)


def verify_node_common_metadata(node):
    """
    Verify pass for metadata at node level
    This pass is used for verification of Metadata. It does sanity checks the metadata of
    each mase node locally, particularly for the following:
    * TODO
    """
    verify_common_metadata_general(node.meta)
    if node.mase_op == "placeholder":
        verify_common_metadata_placeholder(node.meta)
    elif node.mase_op == "output":
        verify_common_metadata_output(node.meta)
    elif node.mase_op == "linear":
        verify_common_metadata_linear(node.meta)
    elif node.mase_op == "relu":
        verify_common_metadata_relu(node.meta)
    elif node.mase_op == "flatten":
        verify_common_metadata_flatten(node.meta)
    else:
        raise ValueError(f"Unknown mase op: {node.mase_op}")


def verify_common_metadata_analysis_pass(graph):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks the common metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO
    """

    # Verify each node int the graph
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
        assert len(node.meta.parameters["common"]["results"]) <= 1

    # Inter-node verification
    # Each edge between nodes must have the same size

    for node in graph.fx_graph.nodes:
        if len(node.all_input_nodes) > 0:
            for i, args in enumerate(node.args):
                assert (
                    node.meta.parameters["common"]["args"][f"data_in_{i}"][
                        "from"
                    ].meta.parameters["common"]["results"][f"data_out_0"]["size"]
                    == node.meta.parameters["common"]["args"][f"data_in_{i}"]["size"]
                )
    return graph


def verify_software_metadata_analysis_pass(graph):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks the software metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO
    """
    return graph


def verify_node_hardware_metadata(node):
    """
    Verify pass for metadata at node level
    This pass is used for verification of Metadata. It does sanity checks the metadata of
    each mase node locally, particularly for the following:
    * TODO
    """
    verify_hardware_metadata_general(node.meta)
    if node.mase_op == "linear":
        verify_hardware_metadata_linear(node.meta)
    elif node.mase_op == "relu":
        verify_hardware_metadata_relu(node.meta)
    else:
        raise ValueError(f"Unknown mase op: {node.op}")


def verify_hardware_metadata_analysis_pass(graph):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks the hardware metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO
    """
    # Verify each node int the graph
    for node in graph.fx_graph.nodes:
        verify_node_hardware_metadata(node)

    # Inter-node verification
    # Each edge between nodes must have the same size
    nodes_in = graph.nodes_in
    nodes_out = graph.nodes_out
    while nodes_in != nodes_out:
        next_nodes_in = []
        for node in nodes_in:
            for next_node, x in node.users.items():
                # This might have a bug - for now assume there is only one result
                if next_node.meta.parameters["hardware"]["is_implicit"]:
                    if node not in next_nodes_in:
                        next_nodes_in.append(node)
                    continue
                next_nodes_in.append(next_node)
                arg_count = len(next_node.all_input_nodes)
                if arg_count == 1:
                    assert (
                        next_node.meta.parameters["hardware"]["verilog_parameters"][
                            "IN_SIZE"
                        ]
                        == node.meta.parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ]
                    ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                        node.name,
                        node.meta.parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ],
                        next_node.name,
                        next_node.meta.parameters["hardware"]["verilog_parameters"][
                            "IN_SIZE"
                        ],
                    )
                else:
                    i = get_input_index(node, next_node)
                    assert (
                        next_node.meta.parameters["hardware"]["verilog_parameters"][
                            f"IN_{i}_SIZE"
                        ]
                        == node.meta.parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ]
                    ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                        node.name,
                        node.meta.parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ],
                        next_node.name,
                        next_node.meta.parameters["hardware"]["verilog_parameters"][
                            f"IN_{i}_SIZE"
                        ],
                    )
        assert (
            nodes_in != next_nodes_in
        ), f"Parsing error: cannot find the next nodes: {nodes_in}."
        nodes_in = next_nodes_in
    return graph


def verify_metadata_analysis_pass(graph):
    """
    Verify pass for mase graph
    This pass is used for verification of MaseGraph. It does sanity checks all the metadata of
    each mase node locally and then verify the inter-node invariants, particularly for the following:
    * TODO
    """
    verify_common_metadata_analysis_pass(graph)
    verify_software_metadata_analysis_pass(graph)
    verify_hardware_metadata_analysis_pass(graph)
    return graph
