from chop.ir.graph.mase_metadata import MaseMetadata
from chop.ir.graph.mase_graph_metadata import MaseGraphMetadata


def init_metadata_analysis_pass(graph, pass_args=None):
    """Initialise a MaseMetadata object
    for each node in the graph

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: arguments for this pass, this pass does not take any argumetns, defaults to None
    :type pass_args: dict, optional
    :return: MaseGraph, pass info (empty in this case)
    :rtype: tuple(MaseGraph, dict)
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"] = MaseMetadata(node=node, model=graph.model)

    # Graph metadata
    graph.fx_graph.meta = {"mase": MaseGraphMetadata(graph)}
    return graph, {}
