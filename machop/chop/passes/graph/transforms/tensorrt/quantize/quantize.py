from copy import copy, deepcopy
import logging
import torch

from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph, get_mase_op,


QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]

def quantize_linear(module):
    if isinstance(module, torch.nn.Linear):

        print(f"Quantizing linear layer: {module}")

def graph_iterator_quantize_by_type(graph, config: dict):
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        b1_graph = None

    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue

        
        
        
    # Apply symbolic tracing to create the FX graph
    traced = torch.fx.symbolic_trace(graph.model)

    # Apply quantization to linear layers in the FX graph
    traced.graph.transform(quantize_linear)

    # Convert the modified FX graph back to a PyTorch model
    quantized_model = traced.to_pytorch_module()

    graph.model = quantized_model

    return graph

    



def graph_iterator_quantize_by_name(graph, config: dict):
    raise NotImplementedError()


def graph_iterator_quantize_by_regex_name(graph, config: dict):
    raise NotImplementedError()


def tensorrt_quantize_transform_pass(graph, pass_args=None):
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_quantize_by_type(graph, pass_args)
        case "name":
            graph = graph_iterator_quantize_by_name(graph, pass_args)
        case "regex_name":
            graph = graph_iterator_quantize_by_regex_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
