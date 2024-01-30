import logging

import toml
import torch
import torch.fx as fx
from chop.ir.graph.mase_metadata import MaseMetadata
from chop.passes.graph.analysis.utils import (
    get_input_nodes,
    get_output_nodes,
)
from torch import nn

from .hardware_metadata_layers import INTERNAL_COMP

logger = logging.getLogger(__name__)

# Here we assume each data has up to three dimensions
MAX_DIM = 3


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def add_component_source(node):
    if node.meta["mase"]["hardware"]["is_implicit"]:
        return

    node.meta["mase"]["hardware"]["interface"] = {}

    mase_op = node.meta["mase"]["common"]["mase_op"]
    if mase_op in INTERNAL_COMP.keys():
        node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL"
        # take the first ip in the component list by default
        node.meta["mase"]["hardware"]["module"] = INTERNAL_COMP[mase_op][0]["name"]
        node.meta["mase"]["hardware"]["dependence_files"] = INTERNAL_COMP[mase_op][0][
            "dependence_files"
        ]
    else:
        node.meta["mase"]["hardware"]["toolchain"] = "HLS"
        node.meta["mase"]["hardware"]["module"] = None
        node.meta["mase"]["hardware"]["dependence_files"] = []

    node.meta["mase"]["hardware"]["device_id"] = -1

    # Current only support on-chip parameters
    args = node.meta["mase"]["common"]["args"]
    for arg, _ in args.items():
        if "data_in" in arg:
            continue
        arg_info = args[arg]
        if isinstance(arg_info, dict):
            node.meta["mase"]["hardware"]["interface"][arg] = {
                "storage": "BRAM",
                "transpose": False,
            }
        else:
            node.meta["mase"]["hardware"]["interface"][arg] = {}


def add_verilog_param(node):
    if node.meta["mase"]["hardware"]["is_implicit"]:
        return

    node.meta["mase"]["hardware"]["verilog_param"] = {}

    args = node.meta["mase"]["common"]["args"]
    results = node.meta["mase"]["common"]["results"]
    vp = node.meta["mase"]["hardware"]["verilog_param"]
    for arg, arg_info in args.items():
        if isinstance(arg_info, dict):
            for i, precision in enumerate(arg_info["precision"]):
                vp[_cap(arg + f"_precision_{i}")] = arg_info["precision"][i]
            for dim in range(0, len(arg_info["shape"])):
                vp[_cap(arg + f"_tensor_size_dim_{dim}")] = (
                    arg_info["shape"][len(arg_info["shape"]) - 1 - dim]
                    if dim < len(arg_info["shape"])
                    else 1
                )
                # If node data parallelism is set, take from hardware metadata
                if node.meta["mase"]["hardware"]["parallelism"] is not None:
                    vp[_cap(arg + f"_parallelism_dim_{dim}")] = node.meta["mase"][
                        "hardware"
                    ]["parallelism"][len(arg_info["shape"]) - 1 - dim]
                # Otherwise, assign to tensor size by default
                else:
                    vp[_cap(arg + f"_parallelism_dim_{dim}")] = (
                        arg_info["shape"][len(arg_info["shape"]) - 1 - dim]
                        if dim < len(arg_info["shape"])
                        else 1
                    )
        elif type(arg_info) == bool:
            vp[_cap(arg)] = 1 if arg_info else 0
        else:
            vp[_cap(arg)] = arg_info

    for result, result_info in node.meta["mase"]["common"]["results"].items():
        if isinstance(result_info, dict):
            for i, precision in enumerate(result_info["precision"]):
                vp[_cap(result + f"_precision_{i}")] = result_info["precision"][i]
            for dim in range(0, len(result_info["shape"])):
                vp[_cap(result + f"_tensor_size_dim_{dim}")] = (
                    result_info["shape"][len(result_info["shape"]) - 1 - dim]
                    if dim < len(result_info["shape"])
                    else 1
                )
                if node.meta["mase"]["hardware"]["parallelism"] is not None:
                    vp[_cap(result + f"_parallelism_dim_{dim}")] = node.meta["mase"][
                        "hardware"
                    ]["parallelism"][len(result_info["shape"]) - 1 - dim]
                else:
                    vp[_cap(result + f"_parallelism_dim_{dim}")] = (
                        result_info["shape"][len(result_info["shape"]) - 1 - dim]
                        if dim < len(result_info["shape"])
                        else 1
                    )
        else:
            vp[_cap(result)] = result_info


def add_hardware_metadata_analysis_pass(graph, pass_args=None):
    """add hardware metadata

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    The hardware metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.

    - hardware
        - is_implicit -> bool : whether the node is mapped on hardware or software annotation only
        - verilog_param -> {} : parameters need for customise the hardware module
        - toolchain -> str : tool chain for code generation, must be INTERNAL, EXTERNAL or HLS
        - module -> str : the name of the used hardware module
        - device_id -> int : the ID of the device where the node is mapped, default = -1
        - interface -> {}
             - name : name of the parameters
                 - storage : the hardware interface implemented, must be BRAM
                 - transpose : whether the data needs to be transposed before emitting
        - dependence_files -> [] : the dependent files for the generated module

    The verilog parameters follow the following naming rules:

    - Hardware signal naming rules

        - Data with tensor types are explicit as hardware signals, such as weight and bias,
          and Data with scalar/tuple types are implicit as parameters (TODO).
        - Each op is a node with a set of inputs, outputs and parameters
        - The input is named by: data_in_0 (data_in_0_ready, data_in_valid), data_in_1,
        - The output is named by: data_out_0 (data_out_0_ready, data_out_valid), data_out_1, ..
        - The parameters are named by PyTorch names: weight (weight_ready, weight_valid), bias (bias_ready, bias_valid)

    - Hardware parameters naming rules
      Parameters with tensor types are explicit as hardware signals, such as weight and bias,
      and parameters with scalar/tuple types are implicit as hardware parameters.

        - Taking data_in_0 for example:
            - `DATA_IN_0_PRECISION_0`
            - `DATA_IN_0_PRECISION_1`
            - ...
            - (depending on how many precision parameters we have.
            - The order matches the same order as the mase precision metadata)
            - `DATA_IN_0_TENSOR_SIZE_DIM_0`
            - `DATA_IN_0_TENSOR_SIZE_DIM_1`
            - `DATA_IN_0_TENSOR_SIZE_DIM_2`
            - `DATA_IN_0_PARALLELISM_DIM_0`
            - `DATA_IN_0_PARALLELISM_DIM_1`
            - `DATA_IN_0_PARALLELISM_DIM_2`
            - (This means that the number of iterations = tensor_size / spatial_size)

        - Implicit parameters are directly translated into verilog parameters, e.g.
          STRIDE
          DIM

    Examples:

    A linear layer in a mase graph with the following common metadata:

    .. code-block:: shell

        %fc1 : [num_users=1] = call_module[target=fc1](args = (%flatten,), kwargs = {})


    .. code-block:: JSON

        {
            "common": {
                "mase_type": "module_related_func",
                "mase_op": "linear",
                "args": {
                    "data_in_0": {
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                        "type": "float",
                        "precision": [32],
                    },
                    "weight": {"type": "float", "precision": [32], "shape": [784, 784]},
                    "bias": {"type": "float", "precision": [32], "shape": [784]},
                },
                "results": {
                    "data_out_0": {
                        "type": "float",
                        "precision": [32],
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                    }
                },
            },
            "software": {},
            "hardware": {},
        }


    The hardware metadata of the linear layer after this pass:

    .. code-block:: JSON

        {
            "common": {...},
            "software": {},
            "hardware": {
                "is_implicit": False,
                "interface": {
                    "weight": {"storage": "BRAM", "transpose": False},
                    "bias": {"storage": "BRAM", "transpose": False},
                },
                "toolchain": "INTERNAL",
                "module": "fixed_linear",
                "device_id": -1,
                "dependence_files": [
                    "cast/fixed_cast.sv",
                    "fixed_arith/fixed_dot_product.sv",
                    "fixed_arith/fixed_vector_mult.sv",
                    "fixed_arith/register_slice.sv",
                    "fixed_arith/fixed_accumulator.sv",
                    "fixed_arith/fixed_adder_tree.sv",
                    "fixed_arith/fixed_adder_tree_layer.sv",
                    "fixed_arith/fixed_mult.sv",
                    "common/join2.sv",
                    "linear/fixed_linear.sv",
                ],
                "verilog_param": {
                    "DATA_IN_0_PRECISION_0": 8,
                    "DATA_IN_0_PRECISION_1": 3,
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 1,
                    "DATA_IN_0_PARALLELISM_DIM_0": 1,
                    "DATA_IN_0_TENSOR_SIZE_DIM_1": 784,
                    "DATA_IN_0_PARALLELISM_DIM_1": 784,
                    "DATA_IN_0_TENSOR_SIZE_DIM_2": 1,
                    "DATA_IN_0_PARALLELISM_DIM_2": 1,
                    "WEIGHT_PRECISION_0": 8,
                    "WEIGHT_PRECISION_1": 3,
                    "WEIGHT_TENSOR_SIZE_DIM_0": 784,
                    "WEIGHT_PARALLELISM_DIM_0": 784,
                    "WEIGHT_TENSOR_SIZE_DIM_1": 784,
                    "WEIGHT_PARALLELISM_DIM_1": 784,
                    "WEIGHT_TENSOR_SIZE_DIM_2": 1,
                    "WEIGHT_PARALLELISM_DIM_2": 1,
                    "BIAS_PRECISION_0": 8,
                    "BIAS_PRECISION_1": 3,
                    "BIAS_TENSOR_SIZE_DIM_0": 784,
                    "BIAS_PARALLELISM_DIM_0": 784,
                    "BIAS_TENSOR_SIZE_DIM_1": 1,
                    "BIAS_PARALLELISM_DIM_1": 1,
                    "BIAS_TENSOR_SIZE_DIM_2": 1,
                    "BIAS_PARALLELISM_DIM_2": 1,
                    "DATA_OUT_0_PRECISION_0": 8,
                    "DATA_OUT_0_PRECISION_1": 3,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_0": 1,
                    "DATA_OUT_0_PARALLELISM_1_DIM_0": 1,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_1": 784,
                    "DATA_OUT_0_PARALLELISM_1_DIM_1": 784,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_2": 1,
                    "DATA_OUT_0_PARALLELISM_1_DIM_2": 1,
                },
            },
        }

    A relu layer in a mase graph with the following common metadata:

    .. code-block:: shell

        %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%fc1,), kwargs = {inplace: False})


    .. code-block:: JSON

        {
            "common": {
                "mase_type": "module_related_func",
                "mase_op": "relu",
                "results": {
                    "data_out_0": {
                        "type": "float",
                        "precision": [32],
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                    }
                },
                "args": {
                    "data_in_0": {
                        "shape": [1, 784],
                        "torch_dtype": torch.float32,
                        "type": "float",
                        "precision": [32],
                    },
                    "inplace": False,
                },
            },
            "software": {},
            "hardware": {},
        }

    The hardware metadata of the relu layer after this pass:

    .. code-block:: JSON

        {
            "common": {...},
            "software": {},
            "hardware": {
                "is_implicit": False,
                "interface": {"inplace": {}},
                "toolchain": "INTERNAL",
                "module": "fixed_relu",
                "device_id": -1,
                "dependence_files": ["activations/fixed_relu.sv"],
                "verilog_param": {
                    "DATA_IN_0_PRECISION_0": 8,
                    "DATA_IN_0_PRECISION_1": 3,
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 1,
                    "DATA_IN_0_PARALLELISM_DIM_0": 1,
                    "DATA_IN_0_TENSOR_SIZE_DIM_1": 784,
                    "DATA_IN_0_PARALLELISM_DIM_1": 784,
                    "DATA_IN_0_TENSOR_SIZE_DIM_2": 1,
                    "DATA_IN_0_PARALLELISM_DIM_2": 1,
                    "INPLACE": False,
                    "DATA_OUT_0_PRECISION_0": 8,
                    "DATA_OUT_0_PRECISION_1": 3,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_0": 1,
                    "DATA_OUT_0_PARALLELISM_1_DIM_0": 1,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_1": 784,
                    "DATA_OUT_0_PARALLELISM_1_DIM_1": 784,
                    "DATA_OUT_0_TENSOR_SIZE_1_DIM_2": 1,
                    "DATA_OUT_0_PARALLELISM_1_DIM_2": 1,
                },
            },
        }

    """

    # Find implicit mase nodes
    for node in graph.nodes:
        node.meta["mase"]["hardware"]["is_implicit"] = False
        node.meta["mase"]["hardware"]["device_id"] = 0

    graph.nodes_in = get_input_nodes(graph.fx_graph)
    graph.nodes_out = get_output_nodes(graph.fx_graph)

    # Add component source
    for node in graph.nodes:
        add_component_source(node)

    # Temporary: fix parallelism to small value to enable verilator simulation
    for node in graph.nodes:
        # Batch parallelism set to 1, data parallelism to 4
        node.meta["mase"]["hardware"]["parallelism"] = [1, 4]

    # Add hardware parameters
    for node in graph.nodes:
        add_verilog_param(node)

    # Add graph metadata
    graph.meta["mase"]["hardware"]["verilog_sources"] = []
    for node in graph.nodes:
        if node.meta["mase"]["hardware"]["is_implicit"]:
            continue
        graph.meta["mase"]["hardware"]["verilog_sources"] += node.meta["mase"][
            "hardware"
        ]["dependence_files"]

    return graph, {}
