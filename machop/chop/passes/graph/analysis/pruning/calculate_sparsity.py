import torch
from chop.passes.graph.analysis.utils import fetch_attr, load_arg
import pdb

def graph_iterator_for_metadata(graph, dummy_in=None, add_value=True):
    """
    largely adapted from https://pytorch.org/docs/stable/fx.html
    """

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
    sparsity_info = {}
    env = {}

    weight_masks = []
    act_masks = []
    for node in graph.fx_graph.nodes:
        args, kwargs = None, None
        if node.op == "placeholder":
            result = dummy_in[node.name]
        elif node.op == "get_attr":
            result = fetch_attr(model, node.target)
        elif node.op == "call_function":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = node.target(*args, **kwargs)
        elif node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = getattr(self_obj, node.target)(*args, **kwargs)
        elif node.op == "call_module":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = modules[node.target](*args, **kwargs)

            meta = node.meta["mase"]
            #if isinstance(modules[node.target], (torch.nn.Conv2d, torch.nn.Linear)):
            if isinstance(modules[node.target], (torch.nn.Conv2d)):
                mask = modules[node.target].parametrizations.weight[0].mask  # weight_mask
                weight_sparsity = 1 - float(mask.sum() / mask.numel())
                meta.parameters["software"]["args"]["weight"][
                    "sparsity"
                ] = weight_sparsity

                act_mask = modules[node.target].activation_mask  # activation
                act_sparsity = 1 - float(act_mask.sum() / act_mask.numel())
                meta.parameters["software"]["args"]["data_in_0"][
                    "sparsity"
                ] = act_sparsity

                if add_value:
                    meta.parameters["software"]["args"]["weight"]["mask_value"] = mask
                    meta.parameters["software"]["args"]["weight_mask"][
                        "value"
                    ] = act_mask
                sparsity_info[node.target] = {
                    "weight_sparsity": weight_sparsity,
                    "activation_sparsity": act_sparsity,
                }

                weight_masks.append(mask)
                act_masks.append(act_mask)

        env[node.name] = result
    return graph, sparsity_info, weight_masks, act_masks


def add_pruning_metadata_analysis_pass(graph, pass_args: dict = {}):
    """
    Add post-pruning metadata analysis pass to the given graph, the graph must have been pruned.

    :param graph: The MaseGraph to which the pruning metadata analysis pass will be added.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the pruning metadata analysis pass.
    This pass requires a dummy_in and a bool value for add_value.
    If add value is true, the mask values would be added to meta data.
    :type pass_args: dict

    :return: The updated graph and sparsity information.
    The returned dict contains {'weight_sparsity': float, 'activation_sparsity': float}
    :rtype: tuple(MaseGraph, dict)
    """

    graph, sparsity_info, weight_masks, act_masks = graph_iterator_for_metadata(
        graph, pass_args["dummy_in"], pass_args["add_value"]
    )
    return graph, sparsity_info, weight_masks, act_masks
