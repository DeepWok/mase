import torch


def fetch_info(node, module):
    # deal with conv2d
    if isinstance(module, torch.nn.Conv2d):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]

        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]

        out = {
            "module_type": "conv2d",
            "weight_value": w_value,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_shape": a_shape,
        }

        # Register weight/activation statistics for pruning methods that require the profile_statistics_analysis_pass
        if "args" in node.meta["mase"].parameters["software"]:
            out["activation_stats"] = node.meta["mase"].parameters["software"]["args"][
                "data_in_0"
            ]["stat"]
            out["weight_stats"] = node.meta["mase"].parameters["software"]["args"][
                "weight"
            ]["stat"]

        return out


def prune_graph_iterator(graph, config: dict):

    # prune in second loop by applying hooks to relevant modules
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            name = node.target

            # remove weights
            if hasattr(graph.modules[node.target], "parametrizations") and hasattr(
                graph.modules[node.target].parametrizations, "weight"
            ):
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
    Apply pruning transformation to the given graph.
    This is achieved by adding a register_parametrization hook to weights
    and a register_pre_forward hook to activations

    :param graph: The input graph to be pruned.
    :type graph: MaseGraph

    pass_args can be None or an empty dictionary.

    :param pass_args: Optional arguments for the pruning transformation.
    :type pass_args: dict

    :return: The pruned graph and an empty dictionary.
    :rtype: tuple
    """
    info = hook_inspector(graph.modules)
    graph = prune_graph_iterator(graph, pass_args)
    info = hook_inspector(graph.modules)
    return graph, {}
