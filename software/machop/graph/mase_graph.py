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

from .dummy_inputs import get_dummy_inputs
from .mase_metadata import MaseMetadata
from .mase_tracer import mase_symbolic_trace
from .utils import get_module_by_name, vf

logger = logging.getLogger(__name__)


def _get_next_call_node(node, nodes_in):
    for next_node, x in node.users.items():
        # No need to synthsize into hardware
        if next_node.target in MaseGraph.implicit_nodes:
            nodes_in = _get_next_call_node(next_node, nodes_in)
            next_node.meta.parameters["hardware"]["is_implicit"] = True
        elif next_node not in nodes_in:
            nodes_in.append(next_node)
    return nodes_in


def _get_prev_call_node(node, nodes_out):
    for prev_node in node.all_input_nodes:
        # No need to synthsize into hardware
        if prev_node.target in MaseGraph.implicit_nodes:
            nodes_out = _get_prev_call_node(prev_node, nodes_out)
            prev_node.meta.parameters["hardware"]["is_implicit"] = True
        elif prev_node not in nodes_out:
            nodes_out.append(prev_node)
    return nodes_out


def _get_input_nodes(fx_graph):
    nodes_in = []
    for node in fx_graph.nodes:
        if node.op == "placeholder":
            nodes_in = _get_next_call_node(node, nodes_in)
            node.meta.parameters["hardware"]["is_implicit"] = True
    return nodes_in


def _get_output_nodes(fx_graph):
    nodes_out = []
    for node in fx_graph.nodes:
        if node.op == "output":
            nodes_out = _get_prev_call_node(node, nodes_out)
            node.meta.parameters["hardware"]["is_implicit"] = True
    return nodes_out


def _list_to_tuple(d):
    """
    Recursively convert list value to tuple value in a nested dict
    """
    for k, v in d.items():
        if isinstance(v, dict):
            _list_to_tuple(v)
        elif isinstance(v, list):
            d[k] = tuple(v)
        else:
            d[k] = v


def get_node_by_name(node_name, fx_graph):
    for node in fx_graph.nodes:
        if vf(node.name) == node_name:
            return node

    assert False, f"Cannot find node {node_name} in fx graph"
    return None


def _add_edge_info(node, fx_graph):
    """
    Set the from metadata to node instead a string
    """
    # TODO: Remove this restriction in the future
    # if node.meta.parameters["hardware"]["is_implicit"]:
    #     return
    arg_count = len(node.all_input_nodes)
    if arg_count != 1:
        for i in range(0, arg_count):
            node_name = node.meta.parameters["common"]["args"][f"data_in_{i}"]["from"]
            node.meta.parameters["common"]["args"][f"data_in_{i}"][
                "from"
            ] = get_node_by_name(node_name, fx_graph)


def get_input_index(node, next_node):
    """
    Get the arg index of the next_node for node
    """
    arg_count = len(next_node.all_input_nodes)
    for i in range(0, arg_count):
        if next_node.meta.parameters["common"]["args"][f"data_in_{i}"]["from"] == node:
            return i


class MaseGraph:
    """
    Mase takes a torch.fx graph representation of a model and translates
    it into a customised representation (Mase graph IR). The Mase graph
    IR is a dataflow representation of the model with both software and
    hardware constraints.
    """

    implicit_nodes = {"size", "view"}
    nonsynthesizable_nodes = {"assert"}

    def __init__(
        self,
        model=None,
        quantized_model=None,
        common_param=None,
        synth_mode="auto",
        args=None,
    ):
        """
        model = input model
        common_param = external common parameters from toml (for quantization info)
        synth_mode = synthesis mode, hls or auto
        """
        self.model = model
        self.quantized_model = quantized_model
        self.synth_mode = synth_mode
        self.args = args
        self.fx_graph = self._init_fx_graph()
        # This has to be before init parameters
        self.nodes_in = _get_input_nodes(self.fx_graph)
        self.nodes_out = _get_output_nodes(self.fx_graph)
        self._init_parameters(common_param=common_param)

    def _init_fx_graph(self):
        model = self.model
        dummy_inputs = get_dummy_inputs(
            model_name=self.args.model, task=self.args.task, model=self.model
        )
        graph_module = mase_symbolic_trace(model, dummy_inputs)
        fx_graph = graph_module.graph
        for node in fx_graph.nodes:
            node.meta = MaseMetadata(
                node=node,
                model=self.model,
                fx_graph=fx_graph,
                quantized_model=self.quantized_model,
                synth_mode=self.synth_mode,
            )
        return fx_graph

    def _init_parameters(self, common_param=None):
        self.init_common_parameters(common_param)
        self.init_software_parameters()
        self.init_hardware_parameters()

    def init_common_parameters(self, load_name):
        if load_name:
            """Update common parameters from a toml file"""
            logging.debug(f"Load common parameters from {load_name}")
            if not load_name.endswith(".toml"):
                raise ValueError("Config file must be a toml file")
            if not os.path.isfile(load_name):
                raise ValueError(
                    f"Config file for quantization not found. Please check if it is provided in the correct path: {load_name}."
                )
            loaded_toml_meta = toml.load(load_name)
            for node in self.fx_graph.nodes:
                if node.op == "call_module" or node.op == "call_function":
                    parameters = loaded_toml_meta[node.name]
                    # !: assign toml to node.meta.parameter
                    _list_to_tuple(parameters)
                    node.meta.parameters["common"] = parameters
                    _add_edge_info(node, self.fx_graph)
                    # TODO: inline the code above into the following API so
                    # verification can be triggered
                    # !: commented out MaseMetadata.init_common_parameters
                    # node.meta.init_common_parameters(parameters=parameters)
        else:
            for node in self.fx_graph.nodes:
                node.meta.init_common_parameters()

    def init_software_parameters(self):
        for node in self.fx_graph.nodes:
            node.meta.init_software_parameters()

    def init_hardware_parameters(self):
        for node in self.fx_graph.nodes:
            node.meta.init_hardware_parameters()

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

        # Each node must have a unique name and a unique verilog name
        node_names = []
        node_vf_names = []
        for node in self.fx_graph.nodes:
            assert node.name not in node_names
            assert vf(node.name) not in node_vf_names
            node_names.append(node.name)
            node_vf_names.append(vf(node.name))

        # Inter-node verification
        # Each edge between nodes must have the same size
        nodes_in = self.nodes_in
        nodes_out = self.nodes_out
        node_in_name = vf(nodes_in[0].name)
        node_out_name = vf(nodes_out[0].name)
        while nodes_in != nodes_out:
            next_nodes_in = []
            for node in nodes_in:
                for next_node, x in node.users.items():
                    arg_count = len(next_node.all_input_nodes)
                    if arg_count == 1:
                        assert (
                            next_node.meta.parameters["common"]["args"]["data_in"][
                                "size"
                            ]
                            == node.meta.parameters["common"]["results"]["data_out"][
                                "size"
                            ]
                        ), "Common input and output sizes mismatch: {} = {} and {} = {}".format(
                            node.name,
                            node.meta.parameters["common"]["results"]["data_out"][
                                "size"
                            ],
                            next_node.name,
                            next_node.meta.parameters["common"]["args"]["data_in"][
                                "size"
                            ],
                        )

                        assert (
                            next_node.meta.parameters["hardware"]["verilog_parameters"][
                                "IN_SIZE"
                            ]
                            == node.meta.parameters["hardware"]["verilog_parameters"][
                                "OUT_SIZE"
                            ]
                        ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                            node.name,
                            node.meta.parameters["hardware"]["verilog_parameters"][
                                "OUT_SIZE"
                            ],
                            next_node.name,
                            next_node.meta.parameters["hardware"]["verilog_parameters"][
                                "IN_SIZE"
                            ],
                        )
                    else:
                        i = get_input_index(node, next_node)
                        assert (
                            next_node.meta.parameters["common"]["args"][f"data_in_{i}"][
                                "size"
                            ]
                            == node.meta.parameters["common"]["results"]["data_out"][
                                "size"
                            ]
                        )
                        assert (
                            next_node.meta.parameters["hardware"]["verilog_parameters"][
                                f"IN_{i}_SIZE"
                            ]
                            == node.meta.parameters["hardware"]["verilog_parameters"][
                                "OUT_SIZE"
                            ]
                        ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                            node.name,
                            node.meta.parameters["hardware"]["verilog_parameters"][
                                "OUT_SIZE"
                            ],
                            next_node.name,
                            next_node.meta.parameters["hardware"]["verilog_parameters"][
                                f"IN_{i}_SIZE"
                            ],
                        )
                    if next_node.op == "output":
                        next_nodes_in.append(node)
                    else:
                        next_nodes_in.append(next_node)
            assert (
                nodes_in != next_nodes_in
            ), f"Parsing error: cannot find the next nodes: {nodes_in}."
            nodes_in = next_nodes_in

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
