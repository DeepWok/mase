#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging
import toml
from pathlib import Path

import torch
import torch.nn as nn

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
    report_node_hardware_type_analysis_pass,
)
from chop.passes.graph.utils import get_mase_op
from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")

class BatchNormNet(nn.Module):
    def __init__(self, channels=64) -> None:
        super().__init__()
        self.net = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.net(x)


class LayerNormNet(nn.Module):
    def __init__(self, chw_shape=[64, 32, 32]) -> None:
        super().__init__()
        self.net = nn.LayerNorm(chw_shape)

    def forward(self, x):
        return self.net(x)


class GroupNormNet(nn.Module):
    def __init__(self, groups=8, chw_shape=[64, 32, 32]) -> None:
        super().__init__()
        self.net = nn.GroupNorm(groups, chw_shape)

    def forward(self, x):
        return self.net(x)


class InstanceNormNet(nn.Module):
    def __init__(self, channels=64) -> None:
        super().__init__()
        self.net = nn.InstanceNorm2d(channels)

    def forward(self, x):
        return self.net(x)


class RMSNormNet(nn.Module):
    def __init__(self, chw_shape=[64, 32, 32]) -> None:
        super().__init__()
        self.net = nn.LayerNorm(chw_shape)

    def forward(self, x):
        return self.net(x)


def _debug_mase_metadata(mg):
    print("#### GRAPH METADATA ####")
    print(mg.meta["mase"].parameters)
    print("#### NODE METADATA ####")
    for n in mg.fx_graph.nodes:
        print(n.meta["mase"].parameters, end="\n\n")

def test_emit_verilog_norm():

    batch_size = 10
    channels = 8
    width = 8
    height = 8

    nn = LayerNormNet(chw_shape=[channels, width, height])
    mg = MaseGraph(model=nn)

    x = torch.randn(batch_size, channels, width, height)
    dummy_in = {"x": x}
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )


    # Remove weight & bias from layernorm (NOT SUPPORTED)
    for n in mg.fx_graph.nodes:
        if get_mase_op(n) == "layer_norm":
            del n.meta["mase"]["common"]["args"]["weight"]
            del n.meta["mase"]["common"]["args"]["bias"]

    # Quantize NN
    quant_config = {
        "by": "type",
        "default": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
            }
        }
    }
    mg, _ = quantize_transform_pass(mg, quant_config)

    # TODO: TEMPORARY PATCH DUE TO QUANTIZER BUG NOT UPDATING METADATA
    for n in mg.fx_graph.nodes:
        common_p = n.meta["mase"].parameters["common"]
        hardware_p = n.meta["mase"].parameters["hardware"]
        for arg, _ in common_p["args"].items():
            if (
                type(common_p["args"][arg]) == dict
                and "type" in common_p["args"][arg].keys()
            ):
                common_p["args"][arg][
                    "type"
                ] = quant_config["default"]["config"]["name"]
                common_p["args"][arg][
                    "precision"
                ] = [
                    quant_config["default"]["config"]["data_in_width"],
                    quant_config["default"]["config"]["data_in_frac_width"]
                ]
        for result, _ in common_p["results"].items():
            if (
                type(common_p["results"][result]) == dict
                and "type"
                in common_p["results"][result].keys()
            ):
                common_p["results"][result][
                    "type"
                ] = quant_config["default"]["config"]["name"]
                common_p["results"][result][
                    "precision"
                ] = [
                    quant_config["default"]["config"]["data_in_width"],
                    quant_config["default"]["config"]["data_in_frac_width"]
                ]
        hardware_p["parallelism"] = [1, 1, 4, 4]

    # Add extra parameters for RTL instantiation
    for n in mg.fx_graph.nodes:
        if get_mase_op(n) == "layer_norm":
            args = n.meta["mase"]["common"]["args"]
            args["INV_SQRT_WIDTH"] = 16
            args["INV_SQRT_FRAC_WIDTH"] = 10
            args["LAYER_NORM"] = 1

    # Add hardware metadata
    mg, _ = add_hardware_metadata_analysis_pass(mg)

    # Emit top level file
    emit_cfg = {
        "project_dir": Path(__file__).parent / "build"
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
    test_emit_verilog_norm()
