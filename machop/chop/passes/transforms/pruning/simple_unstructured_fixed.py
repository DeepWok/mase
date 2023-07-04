import torch

from chop.passes.utils import (
    get_mase_op, get_mase_type, node_actual_target)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]



def apply_parameter_pruning(module, param_name, mask):
    original_weight = getattr(module, param_name)

    module.register_parameter(f'{param_name}_original', original_weight)
    del module._parameters[param_name]

    module.register_buffer(f'{param_name}_mask', mask)
    setattr(module, param_name, original_weight * mask)


def apply_input_pruning(module, hook_fn):
    module.register_forward_pre_hook(hook_fn)
    return module


def hook_fn(module, inputs, outputs):
    threshold = torch.quantile(outputs.abs().flatten(), (1-module.activation_sparsity))
    activation_mask = outputs.abs() > threshold
    activation_mask = activation_mask.to(dtype=outputs.dtype)
    return outputs * activation_mask


def simple_unstructured_fixed_pruning(module: torch.nn.Module, name: str, weight_sparsity: float, activation_sparsity: float):
    if name not in ['weight', 'both', 'activation']:
        raise ValueError(f'Invalid name {name} for simple unstructured fixed pruning.')

    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
        if name in ['weight', 'both']:
            weight = module.weight.data
            threshold = torch.quantile(weight.abs().flatten(), (1-weight_sparsity))
            weight_mask = weight.abs() > threshold 
            weight_mask = weight_mask.to(dtype=weight.dtype)
            apply_parameter_pruning(module, 'weight', weight_mask)
        if name in ['activation', 'both']:
            setattr(module, 'activation_sparsity', activation_sparsity)
            apply_input_pruning(module, hook_fn)
    else:
        raise NotImplementedError(f'Pruning for {type(module)} is not implemented yet.')


PRUNEABLE_OP = ['conv1d', 'conv2d', 'linear']


def graph_iterator_prune(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in PRUNEABLE_OP:
            continue
        if get_mase_type(node) == "module":
            simple_unstructured_fixed_pruning(
                node_actual_target(node), 
                config['name'], 
                config['weight_sparsity'], 
                config['activation_sparsity'])
    return graph


def prune_transform_pass(graph, pass_args=None):
    graph = graph_iterator_prune(graph, pass_args)
    return graph
