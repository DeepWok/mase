#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
from os import makedirs
from pathlib import Path

import torch
import torch.nn as nn

from mase_components.fixed_arithmetic.test.isqrt_sw import make_lut
from mase_components.common.test.lut_tb import write_memb

import chop.models.manual.rms_norm as rms


def _debug_mase_metadata(mg):
    print("#### GRAPH METADATA ####")
    print(mg.meta["mase"].parameters)
    print("#### NODE METADATA ####")
    for n in mg.fx_graph.nodes:
        print(n.meta["mase"].parameters, end="\n\n")


def _fix_quantize_step(node, config={}, parallelism=[1, 1, 2, 2]):
    """
    This function is only required right now because quantize_transform_pass
    is broken and does not assign metadata correctly.
    """

    common_p = node.meta["mase"].parameters["common"]
    hardware_p = node.meta["mase"].parameters["hardware"]
    for arg, _ in common_p["args"].items():
        if (
            type(common_p["args"][arg]) == dict
            and "type" in common_p["args"][arg].keys()
        ):
            common_p["args"][arg][
                "type"
            ] = config["default"]["config"]["name"]
            common_p["args"][arg][
                "precision"
            ] = [
                config["default"]["config"]["data_in_width"],
                config["default"]["config"]["data_in_frac_width"]
            ]
    for result, _ in common_p["results"].items():
        if (
            type(common_p["results"][result]) == dict
            and "type"
            in common_p["results"][result].keys()
        ):
            common_p["results"][result][
                "type"
            ] = config["default"]["config"]["name"]
            common_p["results"][result][
                "precision"
            ] = [
                config["default"]["config"]["data_out_width"],
                config["default"]["config"]["data_out_frac_width"]
            ]
    hardware_p["parallelism"] = parallelism


def add_norm_metadata_gen_lut_analysis_pass(mg, config={}):
    """
    This is a new pass which can be added to MASE that deals with assigning
    metadata to quantized normalization layers and also generates the necessary
    LUTs which exist in the normalization hardware.
    """
    from chop.passes.graph.utils import get_mase_op
    # Generate lut
    LUT_POW = 5
    ISQRT_WIDTH = 16

    mem_dir = Path(__file__).parent / "build" / "norm" / "mem"
    makedirs(mem_dir, exist_ok=True)
    lut = make_lut(2 ** LUT_POW, ISQRT_WIDTH)
    mem_path = mem_dir / f"norm_isqrt_lut.mem"
    write_memb(mem_path, lut, ISQRT_WIDTH)

    # Add extra parameters for RTL instantiation
    for n in mg.fx_graph.nodes:
        mase_op = get_mase_op(n)
        args = n.meta["mase"]["common"]["args"]
        if mase_op == "batch_norm2d":
            # ...
            args["NORM_TYPE"] = "BATCH_NORM"
        elif mase_op == "layer_norm":
            args["ISQRT_LUT_MEMFILE"] = str(mem_path)
            args["NORM_TYPE"] = "LAYER_NORM"
        elif mase_op == "group_norm":
            args["ISQRT_LUT_MEMFILE"] = str(mem_path)
            args["NORM_TYPE"] = "GROUP_NORM"
        elif mase_op == "instance_norm2d":
            args["ISQRT_LUT_MEMFILE"] = str(mem_path)
            args["NORM_TYPE"] = "INSTANCE_NORM"
        elif mase_op == "rms_norm":
            args["ISQRT_LUT_MEMFILE"] = str(mem_path)
            args["NORM_TYPE"] = "RMS_NORM"

    return mg, {}


def test_emit_verilog_norm(net, x):

    import chop.ir.graph.mase_graph as mase_graph
    mg = mase_graph.MaseGraph(model=net)

    from chop.passes.graph.analysis import (
        init_metadata_analysis_pass,
        add_common_metadata_analysis_pass,
        add_hardware_metadata_analysis_pass,
    )
    from chop.passes.graph.transforms import (
        emit_verilog_top_transform_pass,
        emit_internal_rtl_transform_pass,
        emit_cocotb_transform_pass,
        quantize_transform_pass,
    )
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": {"x": x}, "add_value": False}
    )

    # Quantize NN
    quant_config = {
        "by": "type",
        "default": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_out_width": 8,
                "data_out_frac_width": 4,
            }
        }
    }
    mg, _ = quantize_transform_pass(mg, quant_config)

    # TODO: TEMPORARY PATCH DUE TO QUANTIZER BUG NOT UPDATING METADATA
    for n in mg.fx_graph.nodes:
        _fix_quantize_step(n, quant_config)

    # Add norm params
    mg, _ = add_norm_metadata_gen_lut_analysis_pass(mg)

    # Add hardware metadata
    mg, _ = add_hardware_metadata_analysis_pass(mg)

    # Emit top level file
    emit_cfg = {
        "project_dir": Path(__file__).parent / "build",
        "trace": True,
    }
    mg, _ = emit_verilog_top_transform_pass(mg, emit_cfg)

    # Copy over internal rtl
    mg, _ = emit_internal_rtl_transform_pass(mg, emit_cfg)

    # Emit testbench
    _debug_mase_metadata(mg)
    mg, _ = emit_cocotb_transform_pass(mg, emit_cfg)

    # Simulate
    testfile = emit_cfg["project_dir"] / "hardware" / "test" / "mase_top_tb"
    os.system(f"python3 {testfile}/test.py")


if __name__ == "__main__":

    # N, C, H, W
    shape = [10, 4, 8, 8]

    normalizations = [
        # nn.BatchNorm2d(
        #     num_features=shape[1],
        # ),
        nn.LayerNorm(
            normalized_shape=shape[1:],
            elementwise_affine=False,
        ),
        # nn.GroupNorm(
        #     num_groups=2,
        #     num_channels=shape[1],
        #     affine=False,
        # ),
        # nn.InstanceNorm2d(
        #     num_features=shape[1],
        #     affine=False,
        # ),
        # rms.RMSNorm(
        #     normalized_shape=shape[1:],
        # ),
    ]

    x = torch.rand(shape)
    for layer in normalizations:

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_layer = layer

            def forward(self, x):
                return self.norm_layer(x)

        net = Net()
        test_emit_verilog_norm(net, x)
