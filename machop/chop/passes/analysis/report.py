import logging

from tabulate import tabulate

logger = logging.getLogger(__name__)


def inspect_node_analysis_pass(graph):
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


def report(graph, file_name=None):
    """Print out an overview of the model in a table."""
    buff = ""
    buff += str(graph.fx_graph)
    count = {
        "placeholder": 0,
        "get_attr": 0,
        "call_function": 0,
        "call_method": 0,
        "call_module": 0,
        "output": 0,
    }
    layer_types = []
    for node in graph.fx_graph.nodes:
        count[node.op] += 1
    buff += f"""Network overview:
{count}
Layer types:
{layer_types}"""
    if file_name is None:
        print(buff)
    else:
        with open(file_name, "w", encoding="utf-8") as outf:
            outf.write(buff)
    return graph
