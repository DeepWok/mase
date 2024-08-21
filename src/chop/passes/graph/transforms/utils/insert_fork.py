import torch


def insert_fork_transform_pass(graph, pass_args={}):
    """Insert hardware-explicit forks into the mase graph

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)
    """

    logger.info("Inserting forks...")

    nodes_to_fork = []
    for node in graph.fx_graph.nodes:
        user_count = 0
        for u in node.users.keys():
            if u.meta["mase"].parameters["hardware"]["is_implicit"]:
                user_count += 1
        if user_count > 1:
            nodes_to_fork.append(node)

    for node in nodes_to_fork:
        graph.fx_graph.inserting_after(node)
        graph.fx_graph.create_node("call_module", torch.nn.Identity)

    return graph, _
