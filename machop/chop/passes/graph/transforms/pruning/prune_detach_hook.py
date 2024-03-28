import torch


def prune_graph_iterator(graph, config: dict):
    # prune in second loop by applying hooks to relevant modules
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            name = node.target
            # remove weights
            if hasattr(graph.modules[node.target], "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(
                    graph.modules[name], "weight"
                )

            if hasattr(graph.modules[node.target], "_forward_hooks"):
                for k, hook in graph.modules[node.target]._forward_pre_hooks.items():
                    if "sparsify_input" in hook.__name__:
                        del graph.modules[node.target]._forward_pre_hooks[k]

    return graph


def hook_inspector(m):
    info = []
    module_name = type(m).__name__
    if hasattr(m, "_forward_hooks"):
        for k, v in m._forward_hooks.items():
            info.append((module_name, k, v.__name__))

    if hasattr(m, "_forward_pre_hooks"):
        for k, v in m._forward_pre_hooks.items():
            info.append((module_name, k, v.__name__))

    if hasattr(m, "_backward_hooks"):
        for k, v in m._backward_hooks.items():
            info.append((module_name, k, v.__name__))
    return info


def prune_detach_hook_transform_pass(graph, pass_args: dict = {}):
    """
    Remove pruning hooks from a graph. This is done by removing parametrizations and
    deleting the forward hooks from the graph.

    :param graph: The input pruned graph.
    :type graph: MaseGraph

    :param pass_args: Optional arguments for the pruning transformation.
    :type pass_args: dict

    :return: The de-pruned graph and an empty dictionary.
    :rtype: tuple
    """
    info = hook_inspector(graph.modules)
    graph = prune_graph_iterator(graph, pass_args)
    info = hook_inspector(graph.modules)
    return graph, {}
