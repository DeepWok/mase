import logging
import os
import pickle

import toml
import torch
import pickle
import torch.fx as fx
from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
from chop.tools.config_load import convert_none_to_str_na, convert_str_na_to_none

logger = logging.getLogger(__name__)


def save_graph_module_ckpt(graph_module: fx.GraphModule, save_path: str) -> None:
    """Save graph as a checkpoint
    :param graph_module: graph module
    :type graph_module: fx.GraphModule
    :param save_path: the directory for saving
    :type save_path: str
    """
    torch.save(graph_module, save_path)


def save_state_dict_ckpt(graph_module: fx.GraphModule, save_path: str, activation_data: dict = None) -> None:
    """
    Save a serialized state dict.
    """
    state_dict = graph_module.state_dict()
    print("activation data", activation_data)
    if activation_data is not None:
        state_dict={"state_dict": state_dict}
        state_dict["activations"]=activation_data
        print("state_dict", state_dict.keys())
    torch.save(state_dict, save_path)

def save_pickle(graph_module: fx.GraphModule, save_path: str) -> None:
    #Loading pickle_not yet implemented
    with open(save_path, 'wb') as file:
        pickle.dump(graph_module, file)


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


def save_node_meta_param_interface_pass(graph, pass_args: str):
    """
    Save a mase graph metadata.parameters to a toml file.
    """
    node_n_meta_param = collect_n_meta_param(graph)
    save_n_meta_param(node_n_meta_param, pass_args)
    return graph


def load_node_meta_param_interface_pass(graph, pass_args: str):
    """
    Load a mase graph metadata.parameters from a toml file.
    """
    node_n_meta_param = load_n_meta_param(pass_args)
    graph = graph_iterator_add_n_meta_param(graph, node_n_meta_param)
    return graph

def save_pickle(graph_module: fx.GraphModule, save_path: str) -> None:
    #Loading pickle_not yet implemented
    with open(save_path, 'wb') as file:
        pickle.dump(graph_module, file)

def save_mase_graph_interface_pass(graph, pass_args: dict = {}):
    """Save a mase graph.

    This saves the graph module as a serialized graph module and metadata.parameters as a toml file.

    Args:
        graph (MaseGraph): mase_graph to save
        pass_args (str): save directory

    Returns:
        MaseGraph: mase_graph
    """
    transformed=None
    if isinstance(pass_args, dict):
        if "activations" in pass_args.keys():
            transformed=pass_args["activations"]
        save_dir=pass_args["save_dir"]
    else:
        save_dir = pass_args
    os.makedirs(save_dir, exist_ok=True)
    graph_module_ckpt = os.path.join(save_dir, "graph_module.mz")
    state_dict_ckpt = os.path.join(save_dir, "state_dict.pt")
    n_meta_param_ckpt = os.path.join(save_dir, "node_meta_param.toml")
    pickle_ckpt = os.path.join(save_dir, "pickle_save.pkl")
    # collect metadata.parameters
    node_n_meta_param = collect_n_meta_param(graph)
    # save metadata.parameters to toml
    save_n_meta_param(node_n_meta_param, n_meta_param_ckpt)
    # reset metadata to empty dict {}
    #print(graph.model.state_dict())
    graph = graph_iterator_remove_metadata(graph)
    # save graph module & state dict
    if transformed is None:
        save_graph_module_ckpt(graph.model, graph_module_ckpt,)
    save_state_dict_ckpt(graph.model, state_dict_ckpt, transformed)
    save_pickle(graph.model, pickle_ckpt)
    # restore metadata.parameters
    graph, _ = init_metadata_analysis_pass(graph)
    graph = graph_iterator_add_n_meta_param(graph, node_n_meta_param)
    logger.info(f"Saved mase graph to {save_dir}")
    return graph, {}

def save_pruned_train_model(model, pass_args, activation: None):
    save_dir = pass_args
    os.makedirs(save_dir, exist_ok=True)
    state_dict_ckpt = os.path.join(save_dir, "train_prune_state_dict.pt")
    save_state_dict_ckpt(model, state_dict_ckpt, activation) 


def load_mase_graph_interface_pass(graph, pass_args: dict = {"load_dir": None}):
    """
    Load the MASE graph interface pass.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Optional arguments for the transformation pass. Default is {'load_dir': None}, load_dir is required.
    :type pass_args: dict

    :return: The transformed graph and an empty dictionary.
    :rtype: tuple(MaseGraph, dic)

    :raises ValueError: If the load directory is not specified.
    """
    load_dir = pass_args.get("load_dir")
    if load_dir is None:
        raise ValueError(f"load dir cannot be {load_dir}")

    if os.path.isdir(load_dir):
        graph_module_ckpt = os.path.join(load_dir, "graph_module.mz")
        n_meta_param_ckpt = os.path.join(load_dir, "node_meta_param.toml")
    else:
        # Handle the case when the load directory is not a directory
        # ...
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
    return graph, {}
