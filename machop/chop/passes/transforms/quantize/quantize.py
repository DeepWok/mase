import torch

from .quantized_funcs import quantized_func_map
from .quantized_modules import quantized_module_map
from .quantizers import quantizer_map
from copy import copy, deepcopy

from .modify import create_new_module, create_new_fn
from ...utils import get_parent_name
from chop.passes.common import MASE_MODULE, MASE_BUILDIN_FUNCS, MASE_IMPLICIT_FUNCS, MASE_MODULE_RELATED_FUNCS

def get_config(config, name):
    if name in config:
        return config[name]['config']
    else:
        return config['default']['config']


def graph_iterator_quantize_by_type(graph, config):
    for node in graph.fx_graph.nodes:
        node_name = node.target
        if node.mase_op == 'linear':
            node_config = get_config(config, 'linear')
        elif node.mase_op == 'conv1d':
            node_config = get_config(config, 'conv1d')
        elif node.mase_op == 'conv2d':
            node_config = get_config(config, 'conv2d')
        elif node.mase_op == 'relu':
            node_config = get_config(config, 'relu')
            # replace the old module with the new one
        if node.mase_op in MASE_MODULE:
            my_module = graph.modules[node_name]
            new_module = create_new_module(my_module, node_config)
            parent_name, name = get_parent_name(node_name)
            setattr(graph.modules[parent_name], name, new_module)
            # do i need this? it seems like graph.modules keeps track of also all named modules
            graph.modules[node_name] = new_module
            # TODO: also replace meta?
        elif node.mase_op in MASE_BUILDIN_FUNCS:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta = copy(node.meta)
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def graph_iterator_quantize_by_name(graph, config):
    for node in graph.fx_graph.nodes:
        if node.mase_op in MASE_MODULE:
            # quantise module
            node_config = get_config(config, node.name)
            print(node.target, node_config)
            my_module = graph.modules[node.target]
            new_module = create_new_module(my_module, node_config)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # do i need this? it seems like graph.modules keeps track of also all named modules
            graph.modules[node.target] = new_module
            # TODO: also replace meta?
    return graph


def quantize_transform_pass(graph, pass_args=None):
    # TODO: add another hyperparameter to choose quantize by name or by type
    graph = graph_iterator_quantize_by_type(graph, pass_args)
    # TODO: add another pass to report the quantization
    return graph