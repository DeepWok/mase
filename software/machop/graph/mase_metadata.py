import os
import time
import logging
import torch
import torch.fx
from torch import nn
from torch.fx import symbolic_trace
from .utils import get_module_by_name
from .utils import vf


class MaseMetadata:
    """
    The metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.
    - Common
      - args -> []
         - name : name of the arg
           - type : type of the arg, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the arg
      - results -> []
         - name : name of the result
           - type : type of the result, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the result
    - Software
         - ???
    - Hardware
      - verilog_parameters -> {} : parameters need for customise the hardware module
      - target -> str : approach for code generation, must be internal, external or HLS
      - module -> str : the name of the used hardware module
      - dependence_files -> [] : the dependent files for the generated module
    ...
    """

    # Hardware dict
    internal_layers = {nn.Linear: "linear", nn.ReLU: "relu"}

    def __init__(self, node=None, model=None):
        # Top-level model
        self.model = model
        # The target layer/module in the model
        self.module = get_module_by_name(model, node.target)
        # The type of the module
        self.type = type(self.module)
        # The fx node of the module in the fx graph of the model
        self.node = node

    def init_parameters(self):
        # Initialise ll kinds of parameters
        self.parameters = {
            "common": {},
            "software": {},
            "hardware": {},
        }
        if self.node.op == "call_module" or self.node.op == "call_function":
            if self.type in self.internal_layers:
                name = self.internal_layers[self.type]
                replace_fn = getattr(self, f"_init_parameters_{name}")
                replace_fn()
            else:
                logging.warning(f"{self.node} is not found in the internal library.")
                self._init_parameters_other()
        else:
            logging.warning(f"Not dealing with node for now: {self.node}")

    # ----------------------------------------------------------
    # Initialise parameters
    # ----------------------------------------------------------
    def _init_parameters_linear(self):
        node_name = vf(self.node.name)

        # Common parameters
        self.parameters["common"]["args"] = {}
        for name, parameter in self.module.named_parameters():
            self.parameters["common"]["args"][name] = {
                "type": "fixed",
                "precision": (32, 0),
                "size": parameter.shape,
            }
        assert hasattr(
            self.module, "in_features"
        ), f"Linear layer {self.node.name} does not have in features."
        assert hasattr(
            self.module, "out_features"
        ), f"Linear layer {self.node.name} does not have out features."
        self.parameters["common"]["args"]["data_in"] = {
            "type": "fixed",
            "precision": (32, 0),
            "size": (
                1,
                self.module.in_features,
            ),
        }
        self.parameters["common"]["results"] = {
            "data_out": {
                "type": "fixed",
                "precision": (32, 0),
                "size": (
                    1,
                    self.module.out_features,
                ),
            }
        }
        # Hardware parameters
        self.parameters["hardware"] = {
            "verilog_parameters": {
                "IN_WIDTH": self.parameters["common"]["args"]["data_in"]["precision"][
                    0
                ],
                "IN_SIZE": 1,
                "IN_DEPTH": 1,
                "WEIGHT_WIDTH": self.parameters["common"]["args"]["weight"][
                    "precision"
                ][0],
                "PARALLELISM": 2,
                "HAS_BIAS": int("bias" in self.parameters["common"]["args"].keys()),
            },
            "target": "INTERNAL",
            "module": "fixed_linear",
            "dependence_files": [
                "common/fixed_dot_product.sv",
                "common/fixed_vector_mult.sv",
                "common/register_slice.sv",
                "common/fixed_adder_tree.sv",
                "common/fixed_adder_tree_layer.sv",
                "common/fixed_mult.sv",
                "common/join2.sv",
                "linear/fixed_linear.sv",
            ],
        }
        self.parameters["hardware"] = {
            "verilog_parameters": {
                "IN_WIDTH": 32,
                "IN_SIZE": 1,
                "IN_DEPTH": 1,
                "WEIGHT_WIDTH": 32,
                "PARALLELISM": 2,
                "HAS_BIAS": 1,
            },
            "target": "INTERNAL",
            "module": "fixed_linear",
            "dependence_files": [
                "common/fixed_dot_product.sv",
                "common/fixed_vector_mult.sv",
                "common/register_slice.sv",
                "common/fixed_adder_tree.sv",
                "common/fixed_adder_tree_layer.sv",
                "common/fixed_mult.sv",
                "common/join2.sv",
                "linear/fixed_linear.sv",
            ],
        }

    def _init_parameters_relu(self):
        node_name = vf(self.node.name)
        # Common parameters
        self.parameters["common"]["args"] = {}
        for name, parameter in self.module.named_parameters():
            self.parameters["common"]["args"][name] = {
                "type": "fixed",
                "precision": (32, 0),
                "size": parameter.shape,
            }

        # Relu does not have in/out features. Try to fetch from the input nodes
        nodes_in = self.node.args
        nodes_out = list(self.node.users.keys())
        assert len(nodes_in) == 1, f"Relu {self.node.name} has {len(nodes_in)} inputs."
        assert (
            len(nodes_out) == 1
        ), f"Relu {self.node.name} has {len(nodes_out)} outputs."
        node_in = nodes_in[0]
        node_out = nodes_out[0]
        in_features = (
            node_in.meta.module.out_features
            if hasattr(node_in.meta.module, "out_features")
            else -1
        )
        out_features = (
            node_out.meta.module.in_features
            if hasattr(node_out.meta.module, "in_features")
            else -1
        )
        if in_features != -1 and out_features != -1:
            assert (
                in_features == out_features
            ), f"Relu's input ({node_in.name}) and output ({node_out.name}) have different features: {in_features}, {out_features}."
        features = max(in_features, out_features)

        self.parameters["common"]["args"]["data_in"] = {
            "type": "fixed",
            "precision": (32, 0),
            "size": (
                1,
                features,
            ),
        }
        self.parameters["common"]["results"] = {
            "data_out": {
                "type": "fixed",
                "precision": (32, 0),
                "size": (
                    1,
                    features,
                ),
            }
        }

        # Hardware parameters
        self.parameters["hardware"] = {
            "verilog_parameters": {
                "IN_SIZE": 1,
                "IN_WIDTH": self.parameters["common"]["args"]["data_in"]["precision"][
                    0
                ],
            },
            "target": "INTERNAL",
            "module": "fixed_relu",
            "dependence_files": ["activations/fixed_relu.sv"],
        }

    def _init_parameters_other(self):
        node_name = vf(self.node.name)
        # Common parameters
        self.parameters["common"]["args"] = {}
        for name, parameter in self.module.named_parameters():
            self.parameters["common"]["args"][name] = {
                "type": "fixed",
                "precision": (32, 0),
                "size": parameter.shape,
            }

        in_features = 0
        if hasattr(self.module, "in_features"):
            in_features = self.module.in_features
        else:
            nodes_in = self.node.args
            assert (
                len(nodes_in) == 1
            ), f"Module {self.node.name} has {len(nodes_in)} inputs."
            node_in = nodes_in[0]
            if hasattr(node_in.meta.module, "out_features"):
                in_features = node_in.meta.module.out_features
        assert in_features, f"Cannot find the in features for module {self.node.name}"

        out_features = 0
        if hasattr(self.module, "out_features"):
            out_features = self.module.out_features
        else:
            nodes_out = list(self.node.users.keys())
            assert (
                len(nodes_out) == 1
            ), f"Module {self.node.name} has {len(nodes_out)} outputs."
            node_out = nodes_out[0]
            if hasattr(node_out.meta.module, "in_features"):
                out_features = node_out.meta.module.in_features
        assert out_features, f"Cannot find the out features for module {self.node.name}"

        self.parameters["common"]["args"] = {
            "data_in": {
                "type": "fixed",
                "precision": (32, 0),
                "size": (
                    1,
                    in_features,
                ),
            }
        }
        self.parameters["common"]["results"] = {
            "data_out": {
                "type": "fixed",
                "precision": (32, 0),
                "size": (
                    1,
                    out_features,
                ),
            }
        }

        # Hardware parameters
        self.parameters["hardware"]["verilog_parameters"] = {}
        self.parameters["hardware"]["target"] = "HLS"
        self.parameters["hardware"]["module"] = node_name
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
        # Verify common parameters
        assert (
            "data_in" in self.parameters["common"]["args"].keys()
        ), f"Cannot find data_in in common.arg parameters. {self.node}"
        assert (
            "data_out" in self.parameters["common"]["results"].keys()
        ), f"Cannot find data_out in common.arg parameters. {self.node}"

        # Verify hardware parameters
        TARGET = self.parameters["hardware"]["target"]
        assert TARGET in [
            "HLS",
            "INTERNAL",
            "EXTERNAL",
        ], f"Invalid parameter target = {TARGET}. {self.node}"

    def _verify_parameters_linear(self):
        # Verify hardware parameters
        IN_WIDTH = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
        assert IN_WIDTH > 0, f"Invalid parameter IN_WIDTH = {IN_WIDTH}. {self.node}"
        IN_SIZE = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
        assert IN_SIZE > 0, f"Invalid parameter IN_SIZE = {IN_SIZE}. {self.node}"
        IN_DEPTH = self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"]
        assert IN_DEPTH > 0, f"Invalid parameter IN_DEPTH = {IN_DEPTH}. {self.node}"
        WEIGHT_WIDTH = self.parameters["hardware"]["verilog_parameters"]["WEIGHT_WIDTH"]
        assert (
            WEIGHT_WIDTH > 0
        ), f"Invalid parameter WEIGHT_WIDTH = {WEIGHT_WIDTH}. {self.node}"
        PARALLELISM = self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        assert (
            PARALLELISM > 0
        ), f"Invalid parameter PARALLELISM = {PARALLELISM}. {self.node}"
        HAS_BIAS = self.parameters["hardware"]["verilog_parameters"]["HAS_BIAS"]
        assert HAS_BIAS in [
            0,
            1,
        ], f"Invalid parameter HAS_BIAS = {HAS_BIAS}. {self.node}"

    def _verify_parameters_relu(self):
        # Verify hardware parameters
        IN_WIDTH = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
        assert IN_WIDTH > 0, f"Invalid parameter IN_WIDTH = {IN_WIDTH}. {self.node}"
        IN_SIZE = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
        assert IN_SIZE > 0, f"Invalid parameter IN_SIZE = {IN_SIZE}. {self.node}"

    def _verify_parameters_other(self):
        return
