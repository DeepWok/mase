import torch

from .load import load_activation_prune_config, load_weight_prune_config
from .pruning_methods import weight_criteria_map, activation_criteria_map

from .sparse_parameterization import FakeSparseWeight, FakeStructuredSparseWeight


def prune_with_a_function(info, fn, sparsity):
    return fn(info, sparsity)


def get_weight_rank_fn(c):
    return weight_criteria_map[c["scope"]][c["granularity"]][c["method"]]


def get_activation_rank_fn(c):
    return activation_criteria_map[c["scope"]][c["granularity"]][c["method"]]


def get_weight_hook(name, info, named_info, w_config: dict):
    # register parameterization
    w_rank_fn = get_weight_rank_fn(w_config)
    value = named_info["value"]
    w_sparsity = named_info["weight_sparsity"]
    register_parameter_name = "weight"
    parameterization = FakeSparseWeight(w_rank_fn(value, info, w_sparsity))
    return (register_parameter_name, parameterization)


def get_activation_hook(name, info, named_info, a_config: dict):
    a_rank_fn = get_activation_rank_fn(a_config)
    a_sparsity = named_info["activation_sparsity"]

    # register forward hook
    def sparsify_input(module, args):
        if len(args) > 1:
            raise ValueError(
                f"{module.__class__.__name__} takes more than 1 argument at inference, the current sparsiy_input pre forward hook only allows one!"
            )
        x = args[0]
        mask = a_rank_fn(x, info, a_sparsity)
        module.activation_mask = mask
        # it seems like the output of this can be a non-tuple thing??
        return x * mask

    return ("register_forward_pre_hook", sparsify_input)


def build_pruning_hooks(info, w_config, a_config):
    named_hooks = {}
    for k, v in info.items():
        if v is not None:
            w_info = {
                "module_type": v["module_type"],
                "weight_sparsity": w_config["sparsity"],
                "value": v["weight_value"],
                "stats": v["weight_stats"],
                "shape": v["weight_shape"],
            }
            # for activations
            a_info = {
                "module_type": v["module_type"],
                "activation_sparsity": a_config["sparsity"],
                "value": v["activation_value"],
                "stats": v["activation_stats"],
                "shape": v["activation_shape"],
            }
            named_hooks[k] = {
                "w_hook": get_weight_hook(k, info, w_info, w_config),
                "a_hook": get_activation_hook(k, info, a_info, a_config),
            }
    return named_hooks


def fetch_info(node, module):
    # deal with conv2d
    if isinstance(module, torch.nn.Conv2d):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_stats = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]

        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_stats = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]
        return {
            "module_type": "conv2d",
            "weight_value": w_value,
            "weight_stats": w_stats,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_stats": a_stats,
            "activation_shape": a_shape,
        }

    # deal with linear
    if isinstance(module, torch.nn.Linear):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_stats = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]

        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_stats = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]
        return {
            "module_type": "linear",
            "weight_value": w_value,
            "weight_stats": w_stats,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_stats": a_stats,
            "activation_shape": a_shape,
        }

    # otherwise we just return None, and this module would be ignore in build_pruning_hooks
    return None


def prune_graph_iterator(graph, config: dict):
    # Setup all pruning-related parameters (incl. basic validation)
    w_config = load_weight_prune_config(config["weight"], graph)
    a_config = load_activation_prune_config(config["activation"], graph)

    # we need to loop twice, the first time is to fetch all necessary information
    # first sloop
    info = {}
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            module = graph.modules[node.target]
            meta = fetch_info(node, module)
            info[node.target] = meta

    # hook building
    hooks = build_pruning_hooks(info, w_config, a_config)

    # prune in second loop by applying hooks to relevant modules
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            name = node.target
            if name in hooks.keys():
                node_hooks = hooks[name]
                # check weight hook, if it exits, apply it
                if node_hooks["w_hook"] is not None:
                    register_name, parameterization = node_hooks["w_hook"]
                    # apply weigh pruning
                    torch.nn.utils.parametrize.register_parametrization(
                        graph.modules[node.target], register_name, parameterization
                    )
                if node_hooks["a_hook"] is not None:
                    register_fn, hook_fn = node_hooks["a_hook"]
                    getattr(graph.modules[node.target], register_fn)(hook_fn)

    return graph


def prune_transform_pass(graph, pass_args: dict = {}):
    """
    Apply pruning transformation to the given graph.
    This is achieved by adding a register_parametrization hook to weights
    and a register_pre_forward hook to activations

    :param graph: The input graph to be pruned.
    :type graph: MaseGraph

    :param pass_args: Optional arguments for the pruning transformation.
    :type pass_args: dict

    :return: The pruned graph and an empty dictionary.
    :rtype: tuple
    """
    graph = prune_graph_iterator(graph, pass_args)
    return graph, {}
