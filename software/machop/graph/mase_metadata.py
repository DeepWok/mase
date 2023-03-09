import os
import time
import logging
import torch
import torch.fx
from torch import nn
from torch.fx import symbolic_trace
from .utils import get_module_by_name
from .utils import vf


# The metadata of a Mase node in a Mase graph describes the constraints of the
# node for any static analysis or possible transformation. The metadata has a
# tree structure, e.g.
# - Common
#   - args -> []
#      - name : name of the arg
#      - type : type of the arg, e.g. fixed point or float
#      - precision : format of the type, e.g. (10, 5)
#      - size : size of the arg
#   - results -> []
#      - name : name of the result
#      - type : type of the result, e.g. fixed point or float
#      - precision : format of the type, e.g. (10, 5)
#      - size : size of the result
# - Software
#      - ???
# - Hardware
#   - verilog_parameters -> {} : parameters need for customise the hardware module
#   - target -> str : approach for code generation, must be internal, external or HLS
#   - module -> str : the name of the used hardware module
#   - signals -> str : the signals used inside the top-level hardware model
#   - top_signals -> str : the signals used by the interface of the top-level hardware model
#   - port_map -> str : the port map of the current module
#   - dependence_files -> [] : the dependent files for the generated module
# ...
class MaseMetadata:
    # Hardware dict
    internal_layers = {nn.Linear: "linear", nn.ReLU: "relu"}

    def __init__(self, node=None, model=None):
        self.model = model
        self.node = node
        self.type = type(get_module_by_name(model, node.target))
        self.parameters = {"common": {}, "software": {}, "hardware": {}}
        if node.op == "call_module" or node.op == "call_function":
            if self.type in self.internal_layers:
                name = self.internal_layers[self.type]
                replace_fn = getattr(self, f"_init_parameters_{name}")
                replace_fn()
            else:
                logging.warning(f"{node} is not found in the internal library.")
                self._init_parameters_other()
        else:
            logging.warning(f"Not dealing with node for now: {node}")

    # ----------------------------------------------------------
    # Initialise parameters
    # ----------------------------------------------------------
    def _init_parameters_linear(self):
        # Hardware parameters
        self.parameters["hardware"]["verilog_parameters"] = {
            "IN_WIDTH": 32,
            "IN_SIZE": 1,
            "IN_DEPTH": 1,
            "W_WIDTH": 32,
            "PARALLELISM": 2,
            "HAS_BIAS": 1,
        }
        self.parameters["hardware"]["target"] = "INTERNAL"
        self.parameters["hardware"]["module"] = "dataflow_linear"
        node_name = vf(self.node.name)
        self.parameters["hardware"][
            "signals"
        ] = f"""
logic [{node_name}_IN_WIDTH-1:0]  {node_name}_data_in        [{node_name}_IN_SIZE-1:0];
logic                             {node_name}_data_in_valid;
logic                             {node_name}_data_in_ready;
logic [{node_name}_OUT_WIDTH-1:0] {node_name}_data_out            [{node_name}_OUT_SIZE-1:0];
logic                             {node_name}_data_out_valid;
logic                             {node_name}_data_out_ready;
"""
        self.parameters["hardware"][
            "top_signals"
        ] = f"""
input  [{node_name}_W_WIDTH-1:0] {node_name}_weights       [{node_name}_W_WIDTH-1:0],
input                            {node_name}_weights_valid,
output                           {node_name}_weights_ready,
"""
        self.parameters["hardware"][
            "port_map"
        ] = f"""
.data_in       ({node_name}_data_in),
.data_in_valid ({node_name}_data_in_valid),
.data_in_ready ({node_name}_data_in_ready),
.weights      ({node_name}_weights),    
.weights_valid({node_name}_weights_valid),
.weights_ready({node_name}_weights_ready),
.data_out      ({node_name}_data_out),    
.data_out_valid({node_name}_data_out_valid),
.data_out_ready({node_name}_data_out_ready),
"""
        self.parameters["hardware"]["dependence_files"] = [
            "common/fixed_dot_product.sv",
            "common/fixed_vector_mult.sv",
            "common/register_slice.sv",
            "common/fixed_adder_tree.sv",
            "common/fixed_adder_tree_layer.sv",
            "common/fixed_mult.sv",
            "common/join2.sv",
            "linear/fixed_linear.sv",
        ]

    def _init_parameters_relu(self):
        self.parameters["hardware"]["verilog_parameters"] = {
            "IN_SIZE": 1,
            "IN_WIDTH": 32,
        }
        self.parameters["hardware"]["target"] = "INTERNAL"
        self.parameters["hardware"]["module"] = "fixed_relu"
        node_name = vf(self.node.name)
        self.parameters["hardware"][
            "signals"
        ] = f"""
logic [{node_name}_IN_WIDTH-1:0]  {node_name}_data_in        [{node_name}_IN_SIZE-1:0];
logic                             {node_name}_data_in_valid;
logic                             {node_name}_data_in_ready;
logic [{node_name}_OUT_WIDTH-1:0] {node_name}_data_out       [{node_name}_OUT_SIZE-1:0];
logic                             {node_name}_data_out_valid;
logic                             {node_name}_data_out_ready;      
"""
        self.parameters["hardware"]["top_signals"] = ""
        self.parameters["hardware"][
            "port_map"
        ] = f"""
.data_in       ({node_name}_data_in),
.data_in_valid ({node_name}_data_in_valid),
.data_in_ready ({node_name}_data_in_ready),
.data_out      ({node_name}_data_out),    
.data_out_valid({node_name}_data_out_valid),
.data_out_ready({node_name}_data_out_ready),
"""
        self.parameters["hardware"]["dependence_files"] = [
            "activations/fixed_relu.sv",
        ]

    def _init_parameters_other(self):
        self.parameters["hardware"]["verilog_parameters"] = {}
        self.parameters["hardware"]["target"] = "HLS"
        node_name = vf(self.node.name)
        self.parameters["hardware"]["module"] = node_name
        # TODO
        self.parameters["hardware"]["signals"] = ""
        # TODO
        self.parameters["hardware"]["top_signals"] = ""
        # TODO
        self.parameters["hardware"]["port_map"] = ""
        self.parameters["hardware"]["dependence_files"] = []

    # ----------------------------------------------------------
    # Verify parameters
    # ----------------------------------------------------------
    def verify(self):
        if self.node.op == "call_module" or self.node.op == "call_function":
            self._verify_parameters_general()
            if self.type in self.internal_layers:
                name = self.internal_layers[self.type]
                replace_fn = getattr(self, f"_verify_parameters_{name}")
                replace_fn()
            else:
                logging.warning(f"{self.node} is not found in the internal library.")
                self._verify_parameters_other()
        else:
            logging.warning(f"Not dealing with node for now: {self.node}")

    def _verify_parameters_general(self):
        TARGET = self.parameters["hardware"]["target"]
        assert TARGET in [
            "HLS",
            "INTERNAL",
            "EXTERNAL",
        ], f"Invalid parameter target = {TARGET}. {node}"

    def _verify_parameters_linear(self):
        # Hardware parameters
        IN_WIDTH = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
        assert IN_WIDTH > 0, f"Invalid parameter IN_WIDTH = {IN_WIDTH}. {node}"
        IN_SIZE = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
        assert IN_SIZE > 0, f"Invalid parameter IN_SIZE = {IN_SIZE}. {node}"
        IN_DEPTH = self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"]
        assert IN_DEPTH > 0, f"Invalid parameter IN_DEPTH = {IN_DEPTH}. {node}"
        W_WIDTH = self.parameters["hardware"]["verilog_parameters"]["W_WIDTH"]
        assert W_WIDTH > 0, f"Invalid parameter W_WIDTH = {W_WIDTH}. {node}"
        PARALLELISM = self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        assert PARALLELISM > 0, f"Invalid parameter PARALLELISM = {PARALLELISM}. {node}"
        HAS_BIAS = self.parameters["hardware"]["verilog_parameters"]["HAS_BIAS"]
        assert HAS_BIAS in [0, 1], f"Invalid parameter HAS_BIAS = {HAS_BIAS}. {node}"

    def _verify_parameters_relu(self):
        # Hardware parameters
        IN_WIDTH = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
        assert IN_WIDTH > 0, f"Invalid parameter IN_WIDTH = {IN_WIDTH}. {node}"
        IN_SIZE = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
        assert IN_SIZE > 0, f"Invalid parameter IN_SIZE = {IN_SIZE}. {node}"

    def _verify_parameters_other(self):
        return
