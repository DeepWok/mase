from chop.passes.metadata import MaseMetadata


def init_metadata_analysis_pass(graph, pass_args=None):
    """
    Initialise a Mase Metadata object for each node in the graph
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"] = MaseMetadata(node=node, model=graph.model)
    return graph
