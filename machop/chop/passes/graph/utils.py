import inspect
import re
import os
from copy import deepcopy
from typing import Tuple

import torch
from pathlib import Path
import copy

def check_func_type(node, my_func):
    return type(node.target) == type(my_func)


def check_func(node, my_func):
    return node.target == my_func


def isinstance_but_not_subclass(my_object, my_class):
    return my_object.__class__ is my_class


def match_a_pattern(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return pattern
    return None


def get_parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def get_module_by_name(model, request_name):
    for name, layer in model.named_modules():
        if name == request_name:
            return layer
    return None


def get_module_by_target(model, target):
    target_atoms = target.split(".")
    attr_itr = model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_node_by_name(graph, request_name):
    for node in graph.nodes:
        if node.name == request_name:
            return node
    raise RuntimeError(f"No node named {request_name} found in graph")


def vf(string):
    """
    Verilog format
    Format to a compatible verilog name
    """
    return string.replace(".", "_").replace(" ", "_")


def v2p(string):
    """
    Variable to Parameter
    """
    return string.upper()


def get_input_index(node, next_node):
    """
    Get the arg index of the next_node for node
    """
    arg_count = len(next_node.all_input_nodes)
    for i in range(0, arg_count):
        if (
            next_node.meta["mase"].parameters["common"]["args"][f"data_in_{i}"]["from"]
            == node
        ):
            return i


def get_mase_op(node):
    return node.meta["mase"].parameters["common"]["mase_op"]


def get_mase_type(node):
    return node.meta["mase"].parameters["common"]["mase_type"]


def get_node_actual_target(node):
    """
    return the actual target of the node
    - for "call_module": return the torch.nn.Module instance
    - for "call_function": return the function
    - for others: return the node.target
    """
    if node.op == "call_module":
        return node.meta["mase"].module
    elif node.op == "call_function":
        return node.target
    else:
        return node.target


def get_similar_node_actual_target(graph, node):
    """
    return the actual target of the node from the given graph by matching the given node name
    """
    if graph is None:
        return None
    for bl_node in graph.fx_graph.nodes:
        if bl_node.name == node.name:
            return get_node_actual_target(bl_node)


def get_node_target_by_name(graph, request_name):
    if graph is None:
        return None
    for node in graph.fx_graph.nodes:
        if node.name == request_name:
            return get_node_actual_target(node)
    raise RuntimeError(f"No node named {request_name} found in graph")


def deepcopy_mase_graph(mase_graph):
    new_graph = deepcopy(mase_graph)
    for new_n, n in zip(new_graph.fx_graph.nodes, mase_graph.fx_graph.nodes):
        #new_n.meta = deepcopy(n.meta)
        new_n.meta = n.meta
    return new_graph


def init_project(project_dir):
    """
    Create project dir tree
    """
    Path(project_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, "software")).mkdir(parents=True, exist_ok=True)

    hardware_dir = Path(os.path.join(project_dir, "hardware"))
    Path(hardware_dir).mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "rtl").mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "sim").mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "test").mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "test" / "mase_top_tb").mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "test").mkdir(parents=True, exist_ok=True)
    Path(hardware_dir / "hls").mkdir(parents=True, exist_ok=True)
