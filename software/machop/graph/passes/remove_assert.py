def remove_nonsynthesizable_nodes_pass(mase_graph):
    """
    This pass removes non-synthesizable nodes in the mase graph before hardware synthesis
    """
    nodes_to_remove = []
    for node in mase_graph.nodes:
        if node.target in MaseGraph.nonsynthesizable_nodes:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        mase_graph.erase_node(node)

    mase_graph.eliminate_dead_code()
    return mase_graph
