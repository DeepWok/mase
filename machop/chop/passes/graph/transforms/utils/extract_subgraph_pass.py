def extract_subgraph_pass(mase_graph, condition):
    """
    Extract subgraph from a mase graph, for hardware partition
    """
    nodes_to_remove = []
    sub_graph = mase_graph().__deepcopy__()

    for node in sub_graph.nodes:
        if condition(node):
            nodes_to_remove.append(node)

    # If there is a use check, then this needs to switch to an advanced algorithm
    for node in nodes_to_remove:
        sub_graph.erase_node(node)

    return sub_graph
