import logging
import math
import os
import time

import torch
import torch.fx
from torch import nn
from torch.fx import symbolic_trace

from .utils import get_module_by_name, vf, v2p

logger = logging.getLogger(__name__)


def iceil(x):
    return int(math.ceil(x))


def clog2(x):
    return iceil(math.log2(x))


class MaseMetadata:
    """
    The metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.
    - common
      - args -> {}
         - $name : name of the arg
           - type : type of the arg, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the arg
           - from : node
      - results -> {}
         - $name : name of the result
           - type : type of the result, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the result
    - software
         - ???
    - hardware
      - is_implicit -> bool : whether the node is mapped on hardware or software annotation only
      - verilog_parameters -> {} : parameters need for customise the hardware module
      - toolchain -> str : tool chain for code generation, must be INTERNAL, EXTERNAL or HLS
      - module -> str : the name of the used hardware module
      - interface_parameters -> {}
         - name : name of the parameters
           - storage : the hardware interface implemented, must be BRAM
           - transpose : whetehr the data needs to be transposed before emitting
      - dependence_files -> [] : the dependent files for the generated module
    ...
    """

    # Hardware dict
    known_types = ["fixed", "float", "NA"]
    known_toolchain = ["INTERNAL", "EXTERNAL", "HLS"]
    known_storage = ["BRAM"]

    def __init__(
        self,
        node=None,
        model=None,
        fx_graph=None,
        quantized_model=None,
        synth_mode="auto",
    ):
        # Top-level model
        self.model = model
        self.quantized_model = quantized_model
        # The target module in the model
        self.module = get_module_by_name(model, node.target)
        # The type of the module
        self.type = type(self.module)
        # The fx node of the module in the fx graph of the model
        self.node = node
        self.graph = fx_graph
        if synth_mode == "hls":
            self.internal_layers = {}
        else:
            self.internal_layers = {nn.Linear: "linear", nn.ReLU: "relu"}
        self.parameters = {
            "common": {},
            "software": {},
            "hardware": {"is_implicit": False},
        }

    def init_common_parameters(self, parameters=None):
        """
        Init common parameters
        """
        if self.type in self.internal_layers:
            name = self.internal_layers[self.type]
            replace_fn = getattr(self, f"_init_common_parameters_{name}")
            replace_fn(parameters)
        else:
            self._init_common_parameters_other(parameters)

    def init_software_parameters(self, parameters=None):
        """
        Init software parameters
        """
        if self.type in self.internal_layers:
            name = self.internal_layers[self.type]
            replace_fn = getattr(self, f"_init_software_parameters_{name}")
            replace_fn(parameters)
        else:
            self._init_software_parameters_other(parameters)

    def init_hardware_parameters(self, parameters=None):
        """
        Init hardware parameters
        """
        # Ignore implicit nodes
        if self.parameters["hardware"]["is_implicit"]:
            return
        if self.type in self.internal_layers:
            name = self.internal_layers[self.type]
            replace_fn = getattr(self, f"_init_hardware_parameters_{name}")
            replace_fn(parameters)
        else:
            self._init_hardware_parameters_other(parameters)

    def update_hardware_parameters(self, parameters=None):
        """
        Update hardware parameters
        """
        # Ignore implicit nodes
        if self.parameters["hardware"]["is_implicit"]:
            return
        if self.type in self.internal_layers:
            name = self.internal_layers[self.type]
            replace_fn = getattr(self, f"_update_hardware_parameters_{name}")
            replace_fn(parameters)
        else:
            self._update_hardware_parameters_other(parameters)

    def verify(self):
        """
        Verify all the parameters
        """
        self._verify_parameters_general()
        if self.type in self.internal_layers:
            name = self.internal_layers[self.type]
            replace_fn = getattr(self, f"_verify_parameters_{name}")
            replace_fn()
        else:
            self._verify_parameters_other()

    def _verify_parameters_general(self):
        """
        Verify general parameters for all the nodes
        """
        # TODO to remove the following:
        if self.parameters["hardware"]["is_implicit"]:
            return

        # Verify common parameters
        arg_count = len(self.node.all_input_nodes)
        for i in range(0, arg_count):
            arg_name = "data_in" if arg_count == 1 else f"data_in_{i}"
            assert (
                arg_name in self.parameters["common"]["args"].keys()
            ), f"Cannot find data_in in common.arg parameters. {self.node}"
        assert (
            "data_out" in self.parameters["common"]["results"].keys()
        ), f"Cannot find data_out in common.results parameters. {self.node}"

        for arg, param in self.parameters["common"]["args"].items():
            ## Valid type
            arg_type = param["type"]
            assert arg_type in self.known_types, f"Unknown type for {arg} : {arg_type}"
            ## Valid size
            arg_size = param["size"]
            assert arg_size, f"Unknown size for {arg} : {arg_size}"
            ## Data width must be greater than frac width
            if arg_type == "fixed":
                for s in param["size"]:
                    assert isinstance(s, int)
                assert isinstance(param["precision"][0], int)
                assert isinstance(param["precision"][1], int)
                assert param["precision"][0] > 0, f"{arg} must have a positive width."
                assert (
                    param["precision"][1] >= 0
                ), f"{arg} cannot have a negative frac width."
                assert (
                    param["precision"][0] >= param["precision"][1]
                ), f"{arg} must have a width greater than the frac width."
            elif arg_type == "float":
                assert (
                    param["precision"][0] == 32 or param["precision"][0] == 64
                ), f"{arg} must have a width of 32 or 64 as float."
            elif arg_type != "NA":
                assert False, f"Unsupported arg type from toml. {param[type]}"
        assert (
            self.parameters["common"]["args"][arg_name]["type"]
            == self.parameters["common"]["results"]["data_out"]["type"]
        ), "Input and out data type must match. "

        for result, param in self.parameters["common"]["results"].items():
            ## Valid type
            result_type = param["type"]
            assert (
                result_type in self.known_types
            ), f"Unknown type for {result} : {result_type}"
            ## Valid size
            result_size = param["size"]
            assert result_size, f"Unknown size for {result} : {result_size}"
            ## Data width must be greater than frac width
            if result_type == "fixed":
                for s in param["size"]:
                    assert isinstance(s, int)
                assert isinstance(param["precision"][0], int)
                assert isinstance(param["precision"][1], int)
                assert (
                    param["precision"][0] > 0
                ), f"{result} must have a positive width."
                assert (
                    param["precision"][1] >= 0
                ), f"{result} cannot have a negative frac width."
                assert (
                    param["precision"][0] >= param["precision"][1]
                ), f"{result} must have a width greater than the frac width."
            elif result_type == "float":
                assert (
                    param["precision"][0] == 32 or param["precision"][0] == 64
                ), f"{result} must have a width of 32 or 64 as float."
            else:
                assert False, f"Unsupported result type from toml. {param[type]}"

        assert (
            self.parameters["common"]["args"][arg_name]["type"]
            == self.parameters["common"]["results"]["data_out"]["type"]
        ), "Input and out data type must match. "

        # Verify hardware parameters
        for name, param in self.parameters["hardware"]["interface_parameters"].items():
            storage_param = param["storage"]
            assert (
                storage_param in self.known_storage
            ), f"Invalid parameter storage = {storage_param} for {name}. {self.node}"
        toolchain = self.parameters["hardware"]["toolchain"]
        assert (
            toolchain in self.known_toolchain
        ), f"Invalid parameter toolchain = {TARGET}. {self.node}"

        for name, param in self.parameters["hardware"]["verilog_parameters"].items():
            assert isinstance(param, int), f"{name} must be int type. {self.node}"

    # ----------------------------------------------------------
    # Linear
    # ----------------------------------------------------------
    def _init_common_parameters_linear(self, parameters):
        self.parameters["common"]["args"] = {}
        for name, parameter in self.module.named_parameters():
            self.parameters["common"]["args"][name] = {
                "type": "float",
                "precision": [32],
                "size": parameter.shape,
            }
        assert hasattr(
            self.module, "in_features"
        ), f"Linear layer {self.node.name} does not have in features."
        assert hasattr(
            self.module, "out_features"
        ), f"Linear layer {self.node.name} does not have out features."
        self.parameters["common"]["args"]["data_in"] = {
            "type": "float",
            "precision": [32],
            "size": (
                1,
                self.module.in_features,
            ),
        }
        self.parameters["common"]["results"] = {
            "data_out": {
                "type": "float",
                "precision": [32],
                "size": (
                    1,
                    self.module.out_features,
                ),
            }
        }
        if parameters:
            self._update_common_parameters_linear(parameters)

    def _update_common_parameters_linear(self, parameters):
        """
        Toml format check. Example toml:
            ["node.name"]
            name = "integer"
            weight_width = 8
            weight_frac_width = 3
            data_in_width = 8
            data_in_frac_width = 5
            bias_width = 8
            bias_frac_width = 5
            data_out_width = 29
            data_out_frac_width = 10
        """

        assert parameters
        expected_keys = [
            "weight_width",
            "weight_frac_width",
            "data_in_width",
            "data_in_frac_width",
            "data_out_width",
            "data_out_frac_width",
            "bias_width",
            "bias_frac_width",
            "name",
        ]
        expected_keys.sort()
        input_keys = list(parameters.keys())
        input_keys.sort()
        if input_keys != expected_keys:
            assert False, f"""
{self.node.name}: Unexpected parameters found for linear,
expect: {expected_keys},
actual keys: {input_keys}"""

        # Update common parameters
        arg_type = parameters["name"]
        for arg, param in self.parameters["common"]["args"].items():
            # TODO: Do we need a type for each arg?
            self.parameters["common"]["args"][arg]["type"] = arg_type
            if arg_type == "fixed":
                self.parameters["common"]["args"][arg]["precision"] = (
                    parameters[f"{arg}_width"],
                    parameters[f"{arg}_frac_width"],
                )
            else:
                assert False, "Unsupported arg type from toml. Only fixed is supported."
        result_type = arg_type
        for result, param in self.parameters["common"]["results"].items():
            # TODO: Do we need a type for each result?
            self.parameters["common"]["results"][result]["type"] = result_type
            if result_type == "fixed":
                self.parameters["common"]["results"][result]["precision"] = (
                    parameters[f"{result}_width"],
                    parameters[f"{result}_frac_width"],
                )
            else:
                assert (
                    False
                ), "Unsupported result type from toml. Only fixed is supported."

    def _init_software_parameters_linear(self, parameters):
        """
        TODO
        """

    def _init_hardware_parameters_linear(self, parameters):
        arg_type = self.parameters["common"]["args"]["data_in"]["type"]

        if arg_type == "fixed":
            self.parameters["hardware"] |= {
                "verilog_parameters": {
                    "IN_WIDTH": self.parameters["common"]["args"]["data_in"][
                        "precision"
                    ][0],
                    "IN_FRAC_WIDTH": self.parameters["common"]["args"]["data_in"][
                        "precision"
                    ][1],
                    "WEIGHT_WIDTH": self.parameters["common"]["args"]["weight"][
                        "precision"
                    ][0],
                    "WEIGHT_FRAC_WIDTH": self.parameters["common"]["args"]["weight"][
                        "precision"
                    ][1],
                    "HAS_BIAS": int("bias" in self.parameters["common"]["args"].keys()),
                    "BIAS_WIDTH": self.parameters["common"]["args"]["bias"][
                        "precision"
                    ][0],
                    "BIAS_FRAC_WIDTH": self.parameters["common"]["args"]["bias"][
                        "precision"
                    ][1],
                    # Sequential by default
                    "IN_SIZE": 1,
                    "IN_DEPTH": math.prod(
                        self.parameters["common"]["args"]["data_in"]["size"]
                    ),
                    "PARALLELISM": self.parameters["common"]["results"]["data_out"][
                        "size"
                    ][1],
                },
                "toolchain": "INTERNAL",
                "module": "fixed_linear",
                "dependence_files": [
                    "cast/fixed_cast.sv",
                    "common/fixed_dot_product.sv",
                    "common/fixed_vector_mult.sv",
                    "common/register_slice.sv",
                    "common/fixed_accumulator.sv",
                    "common/fixed_adder_tree.sv",
                    "common/fixed_adder_tree_layer.sv",
                    "common/fixed_mult.sv",
                    "common/join2.sv",
                    "linear/fixed_linear.sv",
                ],
            }

            # WEIGHT_SIZE == IN_SIZE * PARALLELISM
            in_size = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
            parallelism = self.parameters["hardware"]["verilog_parameters"][
                "PARALLELISM"
            ]
            self.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"] = (
                in_size * parallelism
            )

            # OUT_WIDTH == IN_WIDTH + WEIGHT_WIDTH + $clog2(IN_SIZE) + $clog2(IN_DEPTH) + HAS_BIAS
            in_width = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
            in_depth = self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"]
            has_bias = self.parameters["hardware"]["verilog_parameters"]["HAS_BIAS"]
            weight_width = self.parameters["hardware"]["verilog_parameters"][
                "WEIGHT_WIDTH"
            ]
            self.parameters["hardware"]["verilog_parameters"]["OUT_WIDTH"] = (
                in_width + weight_width + clog2(in_size) + clog2(in_depth) + has_bias
            )

            # OUT_FRAC_WIDTH == IN_FRAC_WIDTH + WEIGHT_FRAC_WIDTH
            in_frac_width = self.parameters["hardware"]["verilog_parameters"][
                "IN_FRAC_WIDTH"
            ]
            weight_frac_width = self.parameters["hardware"]["verilog_parameters"][
                "WEIGHT_FRAC_WIDTH"
            ]
            self.parameters["hardware"]["verilog_parameters"]["OUT_FRAC_WIDTH"] = (
                in_frac_width + weight_frac_width
            )

            # OUT_SIZE == PARALLELISM
            self.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"] = parallelism
            # BIAS_SIZE == PARALLELISM
            self.parameters["hardware"]["verilog_parameters"]["BIAS_SIZE"] = parallelism

        else:
            assert (
                False
            ), f"Unsupported arg type from toml. Only fixed is supported : {arg_type}"

        self.parameters["hardware"]["interface_parameters"] = {}
        for name, parameter in self.module.named_parameters():
            self.parameters["hardware"]["interface_parameters"][name] = {
                "storage": "BRAM",
                "transpose": False,
            }
        self.parameters["hardware"]["interface_parameters"]["weight"][
            "transpose"
        ] = True

        if parameters:
            self._update_hardware_parameters_linear(parameters)

    def _update_hardware_parameters_linear(self, parameters):
        if parameters is None:
            for pre_node in self.node.all_input_nodes:
                in_size = pre_node.meta.parameters["hardware"]["verilog_parameters"][
                    "OUT_SIZE"
                ]
                self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] = in_size
                total_size = math.prod(
                    self.parameters["common"]["args"]["data_in"]["size"]
                )
                assert total_size % in_size == 0
                self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] = int(
                    total_size / in_size
                )
        else:
            for param, value in parameters.items():
                self.parameters["hardware"]["verilog_parameters"][param] = value
                if param == "IN_SIZE":
                    in_size = math.prod(
                        self.parameters["common"]["args"]["data_in"]["size"]
                    )
                    assert in_size % value == 0
                    self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] = int(
                        in_size / value
                    )
                if param == "IN_DEPTH":
                    in_size = math.prod(
                        self.parameters["common"]["args"]["data_in"]["size"]
                    )
                    assert in_size % value == 0
                    self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] = int(
                        in_size / value
                    )

        # WEIGHT_SIZE == IN_SIZE * PARALLELISM
        in_size = self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
        parallelism = self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        self.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"] = (
            in_size * parallelism
        )

    def _verify_parameters_linear(self):
        # Verify common parameters
        assert (
            self.parameters["common"]["args"]["data_in"]["size"][1]
            == self.parameters["common"]["args"]["weight"]["size"][1]
        ), "Input size does not match with the weight row size. in = {}, w = {}".format(
            self.parameters["common"]["args"]["data_in"]["size"][1],
            self.parameters["common"]["args"]["weight"]["size"][1],
        )
        assert (
            self.parameters["common"]["results"]["data_out"]["size"][1]
            == self.parameters["common"]["args"]["weight"]["size"][0]
        ), "Output size does not match with the weight column size. out = {}, w = {}".format(
            self.parameters["common"]["results"]["data_out"]["size"][1],
            self.parameters["common"]["args"]["weight"]["size"][0],
        )
        if self.parameters["common"]["args"]["data_in"]["type"] == "fixed":
            # Check the output precision based on the input precision - assume lossless
            # Output width = max(bias_width, data_in_width + weight_width + clog2(in_size)) + 1
            # Output frac width = max(bias_frac_width, data_in_frac_width + weight_frac_width)
            bias_width = self.parameters["common"]["args"]["bias"]["precision"][0]
            weight_width = self.parameters["common"]["args"]["weight"]["precision"][0]
            data_in_width = self.parameters["common"]["args"]["data_in"]["precision"][0]
            clog2_data_in_size = clog2(
                self.parameters["common"]["args"]["data_in"]["size"][1]
            )
            bias_frac_width = self.parameters["common"]["args"]["bias"]["precision"][1]
            weight_frac_width = self.parameters["common"]["args"]["weight"][
                "precision"
            ][1]
            data_in_frac_width = self.parameters["common"]["args"]["data_in"][
                "precision"
            ][1]
            expected_precision = (
                max(bias_width, weight_width + data_in_width + clog2_data_in_size) + 1,
                max(bias_frac_width, data_in_frac_width + weight_frac_width),
            )
            assert (
                self.parameters["common"]["results"]["data_out"]["precision"]
                == expected_precision
            ), "Output precision does not match with the estimated precision = {}. Expected = {}".format(
                self.parameters["common"]["results"]["data_out"]["precision"],
                expected_precision,
            )

        # Verify hardware parameters
        data_in_param = self.parameters["common"]["args"]["data_in"]
        if data_in_param["type"] == "fixed":
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
                == data_in_param["precision"][0]
            )
            assert isinstance(data_in_param["precision"][0], int)
            assert isinstance(
                self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"], int
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_FRAC_WIDTH"]
                == data_in_param["precision"][1]
            )
            weight_param = self.parameters["common"]["args"]["weight"]
            assert (
                self.parameters["hardware"]["verilog_parameters"]["WEIGHT_WIDTH"]
                == weight_param["precision"][0]
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["WEIGHT_FRAC_WIDTH"]
                == weight_param["precision"][1]
            )
            bias_param = self.parameters["common"]["args"]["bias"]
            assert (
                self.parameters["hardware"]["verilog_parameters"]["BIAS_WIDTH"]
                == bias_param["precision"][0]
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["BIAS_FRAC_WIDTH"]
                == bias_param["precision"][1]
            )
            assert self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] > 0
            assert self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] > 0
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"]
                * self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
                == data_in_param["size"][1]
            )
            assert self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"] > 0
            assert self.parameters["hardware"]["verilog_parameters"]["HAS_BIAS"] in [
                0,
                1,
            ], f"Invalid parameter HAS_BIAS = {HAS_BIAS}. {self.node}"
            # WEIGHT_SIZE == IN_SIZE * PARALLELISM
            assert (
                self.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"]
                == self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
                * self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
            )
            # OUT_WIDTH == IN_WIDTH + WEIGHT_WIDTH + $clog2(IN_SIZE) + $clog2(IN_DEPTH) + HAS_BIAS
            assert (
                self.parameters["common"]["results"]["data_out"]["precision"][0]
                == self.parameters["hardware"]["verilog_parameters"]["OUT_WIDTH"]
            ), "Output width missmatch for {}, out = {}, expected = {}".format(
                self.node.name,
                self.parameters["hardware"]["verilog_parameters"]["OUT_WIDTH"],
                self.parameters["common"]["results"]["data_out"]["precision"][0],
            )
            # OUT_FRAC_WIDTH == IN_FRAC_WIDTH + WEIGHT_FRAC_WIDTH
            assert (
                self.parameters["common"]["results"]["data_out"]["precision"][1]
                == self.parameters["hardware"]["verilog_parameters"]["OUT_FRAC_WIDTH"]
            )
            # OUT_SIZE == PARALLELISM
            assert (
                self.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"]
                == self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
            )
            # BIAS_SIZE == PARALLELISM
            assert (
                self.parameters["hardware"]["verilog_parameters"]["BIAS_SIZE"]
                == self.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
            )
        else:
            assert False, "Unsupported arg type from toml. Only fixed is supported."

    # ----------------------------------------------------------
    # ReLU
    # ----------------------------------------------------------
    def _init_common_parameters_relu(self, parameters):
        node_name = vf(self.node.name)
        # Common parameters
        self.parameters["common"]["args"] = {}
        # TEMP: Relu does not have in/out features. Try to fetch from the input nodes
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
            "size": (1, features),
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
        if parameters:
            self._update_common_parameters_relu(parameters)

    def _update_common_parameters_relu(self, parameters):
        """
        Toml format check. Example toml:
            ["node.name"]
            name = "integer"
            data_in_width = 8
            data_in_frac_width = 5
            data_out_width = 8
            data_out_frac_width = 5
        """

        assert parameters
        expected_keys = [
            "data_in_width",
            "data_in_frac_width",
            "data_out_width",
            "data_out_frac_width",
            "name",
        ].sort()
        input_keys = list(parameters.keys()).sort()
        if input_keys != expected_keys:
            assert False, f"""
{node_name}: Unexpected parameters found for linear,
expect: {expected_keys},
actual keys: {input_keys}"""

        # Update common parameters
        arg_type = parameters["name"]
        for arg, param in self.parameters["common"]["args"].items():
            # TODO: Do we need a type for each arg?
            self.parameters["common"]["args"][arg]["type"] = arg_type
            if arg_type == "fixed":
                self.parameters["common"]["args"][arg]["precision"] = (
                    parameters[f"{arg}_width"],
                    parameters[f"{arg}_frac_width"],
                )
            else:
                assert False, "Unsupported arg type from toml. Only fixed is supported."
        result_type = arg_type
        for result, param in self.parameters["common"]["results"].items():
            # TODO: Do we need a type for each result?
            self.parameters["common"]["results"][result]["type"] = result_type
            if result_type == "fixed":
                self.parameters["common"]["results"][result]["precision"] = (
                    parameters[f"{result}_width"],
                    parameters[f"{result}_frac_width"],
                )
            else:
                assert (
                    False
                ), "Unsupported result type from toml. Only fixed is supported."

    def _init_software_parameters_relu(self, parameters):
        """
        TODO
        """

    def _init_hardware_parameters_relu(self, parameters):
        if self.parameters["common"]["args"]["data_in"]["type"] == "fixed":
            self.parameters["hardware"] |= {
                "verilog_parameters": {
                    "IN_SIZE": 1,
                    "IN_FRAC_WIDTH": self.parameters["common"]["args"]["data_in"][
                        "precision"
                    ][1],
                    "IN_WIDTH": self.parameters["common"]["args"]["data_in"][
                        "precision"
                    ][0],
                },
                "toolchain": "INTERNAL",
                "module": "fixed_relu",
                "dependence_files": ["activations/fixed_relu.sv"],
            }
        else:
            assert False, "Unsupported arg type from toml. Only fixed is supported."

        # OUT = IN
        self.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"] = self.parameters[
            "hardware"
        ]["verilog_parameters"]["IN_SIZE"]
        self.parameters["hardware"]["verilog_parameters"][
            "OUT_WIDTH"
        ] = self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
        self.parameters["hardware"]["verilog_parameters"][
            "OUT_FRAC_WIDTH"
        ] = self.parameters["hardware"]["verilog_parameters"]["IN_FRAC_WIDTH"]

        self.parameters["hardware"]["interface_parameters"] = {}
        if parameters:
            self._update_hardware_parameters_relu(parameters)

    def _update_hardware_parameters_relu(self, parameters):
        if parameters is None:
            for pre_node in self.node.all_input_nodes:
                if pre_node.op != "call_module" and pre_node.op != "call_function":
                    continue
                in_size = pre_node.meta.parameters["hardware"]["verilog_parameters"][
                    "OUT_SIZE"
                ]
                self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] = in_size
                total_size = math.prod(
                    self.parameters["common"]["args"]["data_in"]["size"]
                )
                assert total_size % in_size == 0
        else:
            for param, value in parameters.items():
                self.parameters["hardware"]["verilog_parameters"][param] = value

        # OUT = IN
        self.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"] = self.parameters[
            "hardware"
        ]["verilog_parameters"]["IN_SIZE"]

    def _verify_parameters_relu(self):
        # Verify common parameters
        assert (
            self.parameters["common"]["args"]["data_in"]
            == self.parameters["common"]["results"]["data_out"]
        ), "ReLU has a mismatched input and output pair"

        # Verify hardware parameters
        data_in_param = self.parameters["common"]["args"]["data_in"]
        if data_in_param["type"] == "fixed":
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
                == data_in_param["precision"][0]
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_FRAC_WIDTH"]
                == data_in_param["precision"][1]
            )
            assert self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] > 0
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_WIDTH"]
                == self.parameters["hardware"]["verilog_parameters"]["OUT_WIDTH"]
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_FRAC_WIDTH"]
                == self.parameters["hardware"]["verilog_parameters"]["OUT_FRAC_WIDTH"]
            )
            assert (
                self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
                == self.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"]
            )
        else:
            assert False, "Unsupported arg type from toml. Only fixed is supported."

    # ----------------------------------------------------------
    # Other
    # ----------------------------------------------------------
    def _init_common_parameters_other(self, parameters):
        self.parameters["common"]["args"] = {}
        self.parameters["common"]["results"] = {}
        if self.module:
            for name, parameter in self.module.named_parameters():
                self.parameters["common"]["args"][name] = {
                    "type": "float",
                    "precision": (32,),
                    "size": parameter.shape,
                }
            for name, parameter in self.module.named_buffers():
                self.parameters["common"]["args"][name] = {
                    "type": "float",
                    "precision": (32,),
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

        arg_count = len(self.node.all_input_nodes)
        if arg_count == 1:
            self.parameters["common"]["args"]["data_in"] = {
                "type": "float",
                "precision": (32,),
                "size": (
                    1,
                    in_features,
                ),
            }
        else:
            for i in range(0, arg_count):
                self.parameters["common"]["args"][f"data_in_{i}"] = {
                    "type": "float",
                    "precision": (32,),
                    "size": (
                        1,
                        in_features,
                    ),
                }

        self.parameters["common"]["results"]["data_out"] = {
            "type": "float",
            "precision": (32,),
            "size": (
                1,
                out_features,
            ),
        }
        if parameters:
            self._update_common_parameters_other(parameters)

    def _update_common_parameters_other(self, parameters):
        assert parameters

        # Update common parameters
        arg_type = parameters["name"]
        for arg, param in self.parameters["common"]["args"].items():
            # TODO: Do we need a type for each arg?
            self.parameters["common"]["args"][arg]["type"] = arg_type
            if arg_type == "fixed":
                self.parameters["common"]["args"][arg]["precision"] = (
                    parameters[f"{arg}_width"],
                    parameters[f"{arg}_frac_width"],
                )
            else:
                assert False, "Unsupported arg type from toml. Only fixed is supported."
        result_type = arg_type
        for result, param in self.parameters["common"]["results"].items():
            # TODO: Do we need a type for each result?
            self.parameters["common"]["results"][result]["type"] = result_type
            if result_type == "fixed":
                self.parameters["common"]["results"][result]["precision"] = (
                    parameters[f"{result}_width"],
                    parameters[f"{result}_frac_width"],
                )
            else:
                assert (
                    False
                ), "Unsupported result type from toml. Only fixed is supported."

        return

    def _init_software_parameters_other(self, parameters):
        """
        TODO
        """

    def _init_hardware_parameters_other(self, parameters):
        node_name = vf(self.node.name)
        self.parameters["hardware"]["toolchain"] = "HLS"
        self.parameters["hardware"]["module"] = node_name
        self.parameters["hardware"]["dependence_files"] = []

        self.parameters["hardware"]["interface_parameters"] = {}
        # Functions and methods do not have named parameters
        if self.module is not None:
            for name, parameter in self.module.named_parameters():
                self.parameters["hardware"]["interface_parameters"][name] = {
                    "storage": "BRAM",
                    "transpose": False,
                }
            for name, parameter in self.module.named_buffers():
                self.parameters["hardware"]["interface_parameters"][name] = {
                    "storage": "BRAM",
                    "transpose": False,
                }

        args_param = self.parameters["common"]["args"]
        results_param = self.parameters["common"]["results"]

        arg_count = len(self.node.all_input_nodes)
        # Add other parameters
        self.parameters["hardware"]["verilog_parameters"] = {}
        for arg, param in args_param.items():
            # NA means not used
            if param["type"] == "NA":
                continue
            cap_arg = v2p(arg)
            if param["type"] == "fixed":
                self.parameters["hardware"]["verilog_parameters"][f"{cap_arg}_SIZE"] = 1
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_arg}_WIDTH"
                ] = param["precision"][0]
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_arg}_FRAC_WIDTH"
                ] = param["precision"][1]
            elif param["type"] == "float":
                self.parameters["hardware"]["verilog_parameters"][f"{cap_arg}_SIZE"] = 1
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_arg}_WIDTH"
                ] = param["precision"][0]
            else:
                assert False, "Unknown type"
        for result, param in results_param.items():
            cap_result = v2p(result)
            if param["type"] == "fixed":
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_result}_SIZE"
                ] = 1
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_result}_WIDTH"
                ] = param["precision"][0]
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_result}_FRAC_WIDTH"
                ] = param["precision"][1]
            elif param["type"] == "float":
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_result}_SIZE"
                ] = 1
                self.parameters["hardware"]["verilog_parameters"][
                    f"{cap_result}_WIDTH"
                ] = param["precision"][0]
            else:
                assert False, "Unknown type"

        if parameters:
            self._update_hardware_parameters_other(parameters)

    def _update_hardware_parameters_other(self, parameters):
        if parameters is None:
            arg_count = len(self.node.all_input_nodes)
            if arg_count == 1:
                for pre_node in self.node.all_input_nodes:
                    in_size = pre_node.meta.parameters["hardware"][
                        "verilog_parameters"
                    ]["OUT_SIZE"]
                    self.parameters["hardware"]["verilog_parameters"][
                        "IN_SIZE"
                    ] = in_size
                    total_size = math.prod(
                        self.parameters["common"]["args"]["data_in"]["size"]
                    )
                    assert (
                        total_size % in_size == 0
                    ), "Size mismatch: {} % {} != 0 {}".format(
                        total_size, in_size, self.node
                    )
                    self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] = int(
                        total_size / in_size
                    )
            else:
                for i in range(0, arg_count):
                    arg_name = f"data_in_{i}"
                    pre_node = self.parameters["common"]["args"][arg_name]["from"]
                    in_size = pre_node.meta.parameters["hardware"][
                        "verilog_parameters"
                    ]["OUT_SIZE"]
                    self.parameters["hardware"]["verilog_parameters"][
                        f"IN_{i}_SIZE"
                    ] = in_size
                    total_size = math.prod(
                        self.parameters["common"]["args"][arg_name]["size"]
                    )
                    assert (
                        total_size % in_size == 0
                    ), "Size mismatch: {} % {} != 0 {}".format(
                        total_size, in_size, self.node
                    )
                    self.parameters["hardware"]["verilog_parameters"][
                        f"IN_{i}_DEPTH"
                    ] = int(total_size / in_size)
        else:
            # TODO: multi-inputs?
            for param, value in parameters.items():
                self.parameters["hardware"]["verilog_parameters"][param] = value
                if param == "IN_SIZE":
                    in_size = math.prod(
                        self.parameters["common"]["args"][f"data_in"]["size"]
                    )
                    assert in_size % value == 0
                    self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] = int(
                        in_size / value
                    )
                elif param.startswith("IN_") and param.endswidth("_SIZE"):
                    index = int(param[param.find("IN_") + 3 : param.find("_SIZE")])
                    in_size = math.prod(
                        self.parameters["common"]["args"][f"data_in_{index}"]["size"]
                    )
                    assert in_size % value == 0
                    self.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] = int(
                        in_size / value
                    )

                if param == "IN_DEPTH":
                    in_size = math.prod(
                        self.parameters["common"]["args"]["data_in"]["size"]
                    )
                    assert in_size % value == 0
                    self.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] = int(
                        in_size / value
                    )

    def _verify_parameters_other(self):
        return
