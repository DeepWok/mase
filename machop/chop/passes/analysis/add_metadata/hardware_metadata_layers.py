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
  - toolchain -> str : tool chain for code generation, must be 
                       INTERNAL_RTL, EXTERNAL_RTL, INTERNAL_HLS, EXTERNAL_HLS, or MLIR_HLS
  - module -> str : the name of the used hardware module
  - interface_parameters -> {}
     - name : name of the parameters
       - storage : the hardware interface implemented, must be BRAM
       - transpose : whetehr the data needs to be transposed before emitting
  - dependence_files -> [] : the dependent files for the generated module
...
"""


import inspect
import math

import torch
from chop.passes.utils import vf

# ----------------------------------------------------------
# Linear
# ----------------------------------------------------------


def analyse_hardware_parameters_linear(meta):
    arg_type = meta.parameters["common"]["args"]["data_in_0"]["type"]

    if arg_type == "fixed":
        meta.parameters["hardware"] |= {
            "verilog_parameters": {
                "HAS_BIAS": int("bias" in meta.parameters["common"]["args"].keys()),
                "IN_0_SIZE": 1,
                "IN_0_DEPTH": math.prod(
                    meta.parameters["common"]["args"]["data_in_0"]["size"]
                ),
                "PARALLELISM": meta.parameters["common"]["results"]["data_out_0"][
                    "size"
                ][1],
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_linear",
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
                # "fixed_arith/fixed_linear.sv",
                "linear/fixed_linear.sv",
            ],
        }

        # WEIGHT_SIZE == IN_SIZE * PARALLELISM
        in_size = meta.parameters["hardware"]["verilog_parameters"]["IN_0_SIZE"]
        parallelism = meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        meta.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"] = (
            in_size * parallelism
        )

        # OUT_SIZE == PARALLELISM
        meta.parameters["hardware"]["verilog_parameters"]["OUT_0_SIZE"] = parallelism
        # BIAS_SIZE == PARALLELISM
        meta.parameters["hardware"]["verilog_parameters"]["BIAS_SIZE"] = parallelism
    elif arg_type == "binary":
        meta.parameters["hardware"] |= {
            "verilog_parameters": {
                "HAS_BIAS": int("bias" in meta.parameters["common"]["args"].keys()),
                "IN_0_SIZE": 1,
                "IN_0_DEPTH": math.prod(
                    meta.parameters["common"]["args"]["data_in_0"]["size"]
                ),
                "PARALLELISM": meta.parameters["common"]["results"]["data_out_0"][
                    "size"
                ][1],
                # Adding precision paramter for binary. TODO: I am not sure if this is the right place for it. If it is we will then add it for the rest of the module
                "IN_0_WIDTH": meta.parameters["common"]["args"]["data_in_0"][
                    "precision"
                ][0],
                "IN_0_FRAC_WIDTH": meta.parameters["common"]["args"]["data_in_0"][
                    "precision"
                ][1],
                "WEIGHT_WIDTH": meta.parameters["common"]["args"]["weight"][
                    "precision"
                ][0],
                "WEIGHT_FRAC_WIDTH": meta.parameters["common"]["args"]["weight"][
                    "precision"
                ][1],
                "BIAS_WIDTH": meta.parameters["common"]["args"]["bias"]["precision"][0],
                "BIAS_FRAC_WIDTH": meta.parameters["common"]["args"]["bias"][
                    "precision"
                ][1],
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_activation_binary_linear",
            # dependence_files for binary acitivation and binary weight
            # "dependence_files": [
            #     "cast/integer_cast.sv",
            #     "cast/fixed_cast.sv",
            #     "linear/binary_activation_binary_linear.sv",
            #     "binary_arith/binary_activation_binary_dot_product.sv",
            #     "fixed_arith/fixed_accumulator.sv",
            #     "binary_arith/binary_activation_binary_vector_mult.sv",
            #     "binary_arith/binary_activation_binary_adder_tree.sv",
            #     "binary_arith/binary_activation_binary_adder_tree_layer.sv",
            #     "binary_arith/binary_activation_binary_mult.sv",
            #     "common/register_slice.sv",
            #     "common/join2.sv",
            # ],
            "dependence_files": [
                "cast/fixed_cast.sv",
                "/binary_arith/fixed_activation_binary_dot_product.sv",
                "/binary_arith/fixed_activation_binary_vector_mult.sv",
                "fixed_arith/register_slice.sv",
                "fixed_arith/fixed_accumulator.sv",
                "fixed_arith/fixed_adder_tree.sv",
                "fixed_arith/fixed_adder_tree_layer.sv",
                "/binary_arith/fixed_activation_binary_mult.sv",
                "common/join2.sv",
                "/linear/fixed_activation_binary_linear.sv",
            ],
        }

        # WEIGHT_SIZE == IN_SIZE * PARALLELISM
        in_size = meta.parameters["hardware"]["verilog_parameters"]["IN_0_SIZE"]
        parallelism = meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        meta.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"] = (
            in_size * parallelism
        )

        # OUT_SIZE == PARALLELISM
        meta.parameters["hardware"]["verilog_parameters"]["OUT_0_SIZE"] = parallelism
        # BIAS_SIZE == PARALLELISM
        meta.parameters["hardware"]["verilog_parameters"]["BIAS_SIZE"] = parallelism
    else:
        meta.parameters["hardware"] |= {
            "verilog_parameters": {},
            "toolchain": "MLIR_HLS",
            "module": vf(meta.node.name),
            "dependence_files": [],
        }

    meta.parameters["hardware"]["interface_parameters"] = {}
    for name, parameter in meta.module.named_parameters():
        meta.parameters["hardware"]["interface_parameters"][name] = {
            "storage": "BRAM",
            "transpose": False,
        }
    meta.parameters["hardware"]["interface_parameters"]["weight"]["transpose"] = True
    return meta


# ----------------------------------------------------------
# ReLU
# ----------------------------------------------------------


def analyse_hardware_parameters_relu(meta):
    # We added binary here, because we believe binary relu should be the same as fixed. And it can be controlled by specifying width and height.
    if (
        meta.parameters["common"]["args"]["data_in_0"]["type"] == "fixed"
        or meta.parameters["common"]["args"]["data_in_0"]["type"] == "binary"
    ):
        meta.parameters["hardware"] |= {
            "verilog_parameters": {
                "IN_0_SIZE": 1,
                # TODO: same here
                "IN_0_WIDTH": meta.parameters["common"]["args"]["data_in_0"][
                    "precision"
                ][0],
                "IN_0_FRAC_WIDTH": meta.parameters["common"]["args"]["data_in_0"][
                    "precision"
                ][1],
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_relu",
            "dependence_files": ["activations/fixed_relu.sv"],
        }
        # OUT = IN
        meta.parameters["hardware"]["verilog_parameters"][
            "OUT_0_SIZE"
        ] = meta.parameters["hardware"]["verilog_parameters"]["IN_0_SIZE"]
    else:
        meta.parameters["hardware"] |= {
            "verilog_parameters": {},
            "toolchain": "MLIR_HLS",
            "module": vf(meta.node.name),
            "dependence_files": [],
        }

    meta.parameters["hardware"]["interface_parameters"] = {}
    return meta
