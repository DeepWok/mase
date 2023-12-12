import torch
from chop.passes.graph.analysis.utils import fetch_attr, load_arg


def graph_iterator(graph, dummy_in, add_meta=False):
    hooks = []
    names, w_infos, a_infos = [], [], []

    # register forward hook
    def get_sparsify(module, args):
        if len(args) > 1:
            raise ValueError(
                f"{module.__class__.__name__} takes more than 1 argument at inference, the current sparsiy_input pre forward hook only allows one!"
            )
        x = args[0]
        a_infos.append((x.numel(), (x != 0).sum() / x.numel()))

    # add hook
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            name = node.target
            if isinstance(graph.modules[name], (torch.nn.Linear, torch.nn.Conv2d)):
                names.append(name)
                graph.modules[name].register_forward_pre_hook(get_sparsify)

    # run it
    env = {}

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
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
            name = node.target
            if isinstance(graph.modules[name], (torch.nn.Linear, torch.nn.Conv2d)):
                w = modules[node.target].weight
                w_infos.append((w.numel(), (w != 0).sum() / w.numel()))
        env[node.name] = result

    w_sparsity_info = {f"{k}_weight": v for k, v in zip(names, w_infos)}
    a_sparsity_info = {f"{k}_activation": v for k, v in zip(names, a_infos)}
    if add_meta:
        for node in graph.fx_graph.nodes:
            if node.op == "call_module":
                name = node.target
                meta = node.meta["mase"]
                if isinstance(modules[node.target], (torch.nn.Conv2d, torch.nn.Linear)):
                    meta.parameters["software"]["args"]["weight"][
                        "natural_sparsity"
                    ] = w_sparsity_info[f"{name}_weight"]
                    meta.parameters["software"]["args"]["data_in_0"][
                        "natural_sparsity"
                    ] = a_sparsity_info[f"{name}_activation"]
    avg_w_sparsity = sum([x[0] * x[1] for x in w_sparsity_info.values()]) / sum(
        [x[0] for x in w_sparsity_info.values()]
    )
    avg_a_sparsity = sum([x[0] * x[1] for x in a_sparsity_info.values()]) / sum(
        [x[0] for x in a_sparsity_info.values()]
    )

    w_sparsity_info["avg_weight"] = avg_w_sparsity
    w_sparsity_info["avg_activation"] = avg_a_sparsity
    return graph, {**w_sparsity_info, **a_sparsity_info}


def add_natural_sparsity_metadata_analysis_pass(graph, pass_args: dict = {}):
    """
    Add natural sparsity metadata analysis pass to the given MaseGraph.
    This is normally used to inspect on the natural sparsity values on both weights and activations.

    :param graph: The MaseGraph to which the analysis pass will be added.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the analysis pass.
    {'dummy_in': tensor, 'add_meta' bool}, add_meta controls whether he natural sparsity would be registered in mase metadata.
    :type pass_args: dict

    :return: The updated MaseGraph and sparsity information.
    The returned dict contains {name (str): sparsity_value (float)}
    :rtype: tuple

    Examples:

    A sample output dict:

    .. code-block:: JSON

        {
            'avg_activation': tensor(0.6709),
            'avg_weight': tensor(1.0000),
            'conv1_activation': (6144, tensor(1.)),
            'conv1_weight': (9408, tensor(1.)),
            'fc_activation': (1024, tensor(0.6289)),
            'fc_weight': (5120, tensor(1.)),
            'layer1.0.conv1_activation': (8192, tensor(0.8810)),
            'layer1.0.conv1_weight': (36864, tensor(1.)),
            ...
        }

    """
    graph, sparsity_info = graph_iterator(
        graph, pass_args["dummy_in"], add_meta=pass_args["add_meta"]
    )
    return graph, sparsity_info
