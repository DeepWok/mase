import logging
import os

import toml
import torch
import torch.fx as fx
from chop.passes.analysis.init_metadata import init_metadata_analysis_pass
from chop.tools.config_load import convert_none_to_str_na, convert_str_na_to_none

logger = logging.getLogger(__name__)


def save_graph_module_ckpt(graph_module: fx.GraphModule, save_path: str) -> None:
    """
    Save a serialized graph module.
    """
    torch.save(graph_module, save_path)


def save_state_dict_ckpt(graph_module: fx.GraphModule, save_path: str) -> None:
    """
    Save a serialized state dict.
    """
    state_dict = graph_module.state_dict()
    torch.save(state_dict, save_path)


def graph_iterator_remove_metadata(graph):
    """
    Remove all metadata from the graph.
    """
    for node in graph.fx_graph.nodes:
        if hasattr(node, "meta"):
            node.meta["mase"] = {}
    return graph


def collect_n_meta_param(graph) -> dict:
    """
    Collect all metadata from the graph.
    """

    node_n_meta_param = {}
    for node in graph.fx_graph.nodes:
        node_n_meta_param[node.name] = node.meta["mase"].parameters
    return node_n_meta_param


def save_n_meta_param(node_meta: dict, save_path: str) -> None:
    """
    Save a mase graph metadata to a toml file.
    """
    node_meta = convert_none_to_str_na(node_meta)
    with open(save_path, "w") as f:
        toml.dump(node_meta, f)


def load_n_meta_param(load_path: str) -> dict:
    """
    Load a mase graph metadata from a toml file.
    """
    with open(load_path, "r") as f:
        node_meta = toml.load(f)
    node_meta = convert_str_na_to_none(node_meta)
    return node_meta


def graph_iterator_add_n_meta_param(graph, node_n_meta_param: dict):
    """
    Add metadata to the graph.
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"].parameters = node_n_meta_param[node.name]
    return graph


def load_graph_module_ckpt(checkpoint: str) -> fx.GraphModule:
    """
    Load a serialized graph module.
    """
    graph_module = torch.load(checkpoint)
    return graph_module


def graph_iterator_add_n_meta_param(graph, node_n_meta_param: dict):
    """
    Add metadata to the graph.
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"].parameters = node_n_meta_param[node.name]
    return graph


def save_node_meta_param_transform_pass(graph, pass_args: str):
    """
    Save a mase graph metadata.parameters to a toml file.
    """
    node_n_meta_param = collect_n_meta_param(graph)
    save_n_meta_param(node_n_meta_param, pass_args)
    return graph


def load_node_meta_param_transform_pass(graph, pass_args: str):
    """
    Load a mase graph metadata.parameters from a toml file.
    """
    node_n_meta_param = load_n_meta_param(pass_args)
    graph = graph_iterator_add_n_meta_param(graph, node_n_meta_param)
    return graph


def save_mase_graph_transform_pass(graph, pass_args: str):
    """Save a mase graph.

    This saves the graph module as a serialized graph module and metadata.parameters as a toml file.

    Args:
        graph (MaseGraph): mase_graph to save
        pass_args (str): save directory

    Returns:
        MaseGraph: mase_graph
    """
    save_dir = pass_args
    os.makedirs(save_dir, exist_ok=True)
    graph_module_ckpt = os.path.join(save_dir, "graph_module.mz")
    state_dict_ckpt = os.path.join(save_dir, "state_dict.pt")
    n_meta_param_ckpt = os.path.join(save_dir, "node_meta_param.toml")
    # collect metadata.parameters
    node_n_meta_param = collect_n_meta_param(graph)
    # save metadata.parameters to toml
    save_n_meta_param(node_n_meta_param, n_meta_param_ckpt)
    # reset metadata to empty dict {}
    graph = graph_iterator_remove_metadata(graph)
    # save graph module & state dict
    save_graph_module_ckpt(graph.model, graph_module_ckpt)
    save_state_dict_ckpt(graph.model, state_dict_ckpt)
    # restore metadata.parameters
    graph = init_metadata_analysis_pass(graph)
    graph = graph_iterator_add_n_meta_param(graph, node_n_meta_param)
    logger.info(f"Saved mase graph to {save_dir}")
    return graph


def load_mase_graph_transform_pass(graph, pass_args: str):
    """Load a mase graph.

    This loads the graph module as a serialized graph module and metadata.parameters as a toml file.

    Args:
        graph (MaseGraph): mase_graph to load
        pass_args (str): load directory

    Returns:
        MaseGraph: mase_graph
    """
    load_dir = pass_args
    if os.path.isdir(load_dir):
        graph_module_ckpt = os.path.join(load_dir, "graph_module.mz")
        n_meta_param_ckpt = os.path.join(load_dir, "node_meta_param.toml")
    else:
        load_dir = os.path.dirname(load_dir)
        graph_module_ckpt = os.path.join(load_dir, "graph_module.mz")
        n_meta_param_ckpt = os.path.join(load_dir, "node_meta_param.toml")
    # load metadata.parameters from toml
    node_n_meta_param = load_n_meta_param(n_meta_param_ckpt)
    # load graph module
    graph.model = load_graph_module_ckpt(graph_module_ckpt)
    graph.model.additional_inputs = {}
    graph = init_metadata_analysis_pass(graph)
    # add metadata.parameters to graph
    graph = graph_iterator_add_n_meta_param(graph, node_n_meta_param)
    logger.info(f"Loaded mase graph from {load_dir}")
    return graph
