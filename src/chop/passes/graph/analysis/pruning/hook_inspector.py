import torch


from chop.passes.graph.analysis.utils import fetch_attr, load_arg


def graph_iterator_for_metadata(graph):
    """
    largely adapted from https://pytorch.org/docs/stable/fx.html
    """

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
    hook_info = {}

    for node in graph.fx_graph.nodes:
        if node.op == "call_module":
            name = node.target
            if isinstance(modules[node.target], (torch.nn.Conv2d, torch.nn.Linear)):
                m = modules[name]
                for k, v in m._forward_hooks.items():
                    hook_info[f"{name}_{k}"] = (k, v)

                for k, v in m._forward_pre_hooks.items():
                    hook_info[f"{name}_{k}"] = (k, v)

                for k, v in m._backward_hooks.items():
                    hook_info[f"{name}_{k}"] = (k, v)

    return graph, hook_info


def hook_inspection_analysis_pass(graph, pass_args: dict = {}):
    """
    Remove and provide hook information of the modules.

    :param graph: The MaseGraph to which the pruning metadata analysis pass will be added.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the pruning metadata analysis pass. This pass does not need any values, so an empty dictionary is fine
    :type pass_args: dict

    :return: The updated graph and sparsity information. The returned dict contains {'module_name': (hook_id, hook_fn)}
    :rtype: tuple(MaseGraph, dict)

    Examples:

    A sample output dict:

    .. code-block:: JSON

        {
            'feature_layers.0_0': (
                0, <function get_activation_hook.<locals>.sparsify_input at 0x7f9544528c10>),
            'feature_layers.3_1': (
                1, <function get_activation_hook.<locals>.sparsify_input at 0x7f9544528ca0>),
            'feature_layers.7_2': (
                2, <function get_activation_hook.<locals>.sparsify_input at 0x7f9544528d30>),
        }

    """

    graph, hook_info = graph_iterator_for_metadata(graph)
    return graph, hook_info
