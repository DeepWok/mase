import glob
import logging
import os
import shutil
import time

import toml
import torch
import torch.fx
from torch import nn
from torch.fx import symbolic_trace

from .mase_metadata import MaseMetadata

logger = logging.getLogger(__name__)


def _get_next_call_node(node, nodes_in):
    for next_node, x in node.users.items():
        if next_node.op != "call_module" and next_node.op != "call_function":
            nodes_in = _get_next_call_node(next_node, nodes_in)
        elif next_node not in nodes_in:
            nodes_in.append(next_node)
    return nodes_in


def _get_prev_call_node(node, nodes_out):
    for prev_node in node.all_input_nodes:
        if prev_node.op != "call_module" and prev_node.op != "call_function":
            nodes_out = _get_prev_call_node(prev_node, nodes_out)
        elif prev_node not in nodes_out:
            nodes_out.append(prev_node)
    return nodes_out


def _get_input_nodes(fx_graph):
    nodes_in = []
    for node in fx_graph.nodes:
        if node.op == "placeholder":
            nodes_in = _get_next_call_node(node, nodes_in)
    return nodes_in


def _get_output_nodes(fx_graph):
    nodes_out = []
    for node in fx_graph.nodes:
        if node.op == "output":
            nodes_out = _get_prev_call_node(node, nodes_out)
    return nodes_out


# Mase takes a torch.fx graph representation of a model and translates
# it into a customised representation (Mase graph IR). The Mase graph
# IR is a dataflow representation of the model with both software and
# hardware constraints.
class MaseGraph:
    def __init__(self, model=None, common_param=None):
        self.model = model
        self.param = param
        self.q_param = q_param
        self.fx_graph, self.nodes_in, self.nodes_out = self._init_fx_graph()
        self._init_parameters(common_param=common_param)

    def _init_fx_graph(self):
        model = self.model
        trace = torch.fx.symbolic_trace(model)
        trace.graph.lint()
        trace = trace
        fx_graph = trace.graph
        nodes_in = _get_input_nodes(fx_graph)
        assert len(nodes_in) == 1, "Multiple inputs are not supported."
        nodes_out = _get_output_nodes(fx_graph)
        assert len(nodes_out) == 1, "Multiple outputs are not supported."
        for node in fx_graph.nodes:
            node.meta = MaseMetadata(node=node, model=model)
        return fx_graph, nodes_in, nodes_out

    def _init_parameters(self, common_param=None):
        self.update_common_parameters(common_param)
        self.update_software_parameters()
        self.update_hardware_parameters()

    def update_common_parameters(self, load_name):
        if load_name:
            """Update common parameters from a toml file"""
            if not load_name.endswith(".toml"):
                raise ValueError("Config file must be a toml file")
            loaded_toml_meta = toml.load(load_name)
            for node in self.fx_graph.nodes:
                parameters = loaded_toml_meta[node.name]
                node.meta.update_common_parameters(parameters=parameters)
        else:
            for node in self.fx_graph.nodes:
                node.meta.update_common_parameters()

    def update_software_parameters(self):
        for node in self.fx_graph.nodes:
            node.meta.update_software_parameters()

    def update_hardware_parameters(self):
        for node in self.fx_graph.nodes:
            node.meta.update_hardware_parameters()

    def load_parameters(self, load_name):
        """Load complete parameters from a toml file"""
        if not load_name.endswith(".toml"):
            raise ValueError("Config file must be a toml file")
        loaded_toml_meta = toml.load(load_name)
        for node in self.fx_graph.nodes:
            node.meta.parameters = loaded_toml_meta[node.name]
        self.verify()

    def save_parameters(self, save_name):
        """Save all the constraints to a toml file"""
        toml_meta_to_save = {}
        for node in self.fx_graph.nodes:
            toml_meta_to_save[node.name] = node.meta.parameters
        with open(save_name, "w") as f:
            toml_meta_string = toml.dump(toml_meta_to_save, f)

    def verify(self):
        # Verify each node itself
        for node in self.fx_graph.nodes:
            node.meta.verify()
        # Inter-node verification
        # Each edge between nodes must have the same precision
        # TODO
        # Each node must have a unique name and a unique verilog name
        # TODO

    def report(self):
        """Print out an overview of the model in a table."""
        count = {
            "placeholder": 0,
            "get_attr": 0,
            "call_function": 0,
            "call_method": 0,
            "call_module": 0,
            "output": 0,
        }
        layer_types = []
        for node in self.fx_graph.nodes:
            count[node.op] += 1
        logger.debug(
            f"""Network overview:
{count}
Layer types:
{layer_types}"""
        )
