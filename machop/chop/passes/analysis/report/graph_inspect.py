import logging

from tabulate import tabulate

logger = logging.getLogger(__name__)


def graph_iterator_inspect_node(graph):
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
    logger.debug("Inspecting graph [add_common_metadata_analysis_pass]")
    logger.debug("\n" + tabulate(rows, headers=headers))
    return graph


def report_metadata_analysis_pass(graph, pass_args=None):
    """Inspect mase graph after initialization/loading, including

    - node inspection
    """
    graph = graph_iterator_inspect_node(graph)
    return graph
