import logging

from tabulate import tabulate

logger = logging.getLogger(__name__)


def graph_iterator_inspect_node_type(graph):
    headers = ["Node name", "Fx Node op", "Mase type", "Mase op"]
    rows = []
    for node in graph.fx_graph.nodes:
        rows.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["mase_type"],
                node.meta["mase"].parameters["common"]["mase_op"],
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
