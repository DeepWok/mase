import logging
from pprint import pformat
from tabulate import tabulate
from pathlib import Path

logger = logging.getLogger(__name__)


def graph_iterator_inspect_node_type(graph):
    headers = ["Node name", "Fx Node op", "Mase type", "Mase op", "Value type"]
    rows = []
    for node in graph.fx_graph.nodes:
        rows.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["mase_type"],
                node.meta["mase"].parameters["common"]["mase_op"],
                "NA"
                if "data_in_0" not in node.meta["mase"].parameters["common"]["args"]
                else node.meta["mase"].parameters["common"]["args"]["data_in_0"][
                    "type"
                ],
            ]
        )
    logger.info("Inspecting graph [add_common_node_type_analysis_pass]")
    logger.info("\n" + tabulate(rows, headers=headers))
    return graph


def report_node_type_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on types
    """
    graph = graph_iterator_inspect_node_type(graph)
    return graph


def graph_iterator_inspect_node_shape(graph):
    logger.info("Inspecting graph [add_common_node_shape_analysis_pass]")
    buffer = ""
    for node in graph.fx_graph.nodes:
        buffer += f"{node.name}:\nin:\n"
        for key, value in node.meta["mase"].parameters["common"]["args"].items():
            from_name = "none" if value["from"] is None else value["from"].name
            buffer += "{} = {}, from = {}\n".format(key, value["size"], from_name)
        buffer += "out:\n"
        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            buffer += "{} = {}\n".format(key, value["size"])
        buffer += "\n"
    logger.info(buffer)
    return graph


def report_node_shape_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on shapes
    """
    graph = graph_iterator_inspect_node_shape(graph)
    return graph


def graph_iterator_inspect_node_hardware_type(graph):
    headers = ["Node name", "Fx Node op", "Type", "Tool Chain"]
    rows = []
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        rows.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["results"]["data_out_0"]["type"],
                node.meta["mase"].parameters["hardware"]["toolchain"],
            ]
        )
    logger.info("Inspecting graph [add_common_node_hardware_type_analysis_pass]")
    logger.info("\n" + tabulate(rows, headers=headers))
    return graph


def report_node_hardware_type_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on hardware types
    """
    graph = graph_iterator_inspect_node_hardware_type(graph)
    return graph


def report_node_meta_param_analysis_pass(graph, pass_args=None):
    headers = [
        "Node name",
        "Fx Node op",
        "Mase type",
        "Mase op",
        "Common Param",
        "Hardware Param",
        "Software Param",
    ]

    row = []
    for node in graph.fx_graph.nodes:
        row.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["mase_type"],
                node.meta["mase"].parameters["common"]["mase_op"],
                pformat(node.meta["mase"].parameters["common"]),
                pformat(node.meta["mase"].parameters["hardware"]),
                pformat(node.meta["mase"].parameters["software"]),
            ]
        )

    table_txt = tabulate(row, headers=headers, tablefmt="grid")
    logger.info("Inspecting graph [add_common_meta_param_analysis_pass]")
    logger.info("\n" + table_txt)
    return graph
