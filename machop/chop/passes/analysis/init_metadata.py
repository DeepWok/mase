from chop.passes.metadata import MaseMetadata


def init_metadata_analysis_pass(graph, pass_args=None):
    for node in graph.fx_graph.nodes:
        node.meta = MaseMetadata(
            node=node,
            model=graph.model,
            fx_graph=graph.fx_graph
        )
    return graph