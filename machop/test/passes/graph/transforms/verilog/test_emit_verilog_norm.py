#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging
import toml

import torch
import torch.nn as nn

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)
from chop.passes.graph.utils import get_mase_op
from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

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
    for n in mg.fx_graph.nodes:
        print(n.meta["mase"].parameters, end="\n\n")

def test_emit_verilog_norm():

    batch_size = 10
    channels = 32
    width = 32
    height = 32

    nn = LayerNormNet(chw_shape=[channels, width, height])
    mg = MaseGraph(model=nn)

    x = torch.randn(batch_size, channels, width, height)
    dummy_in = {"x": x}
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

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

    mg, _ = add_hardware_metadata_analysis_pass(mg)

    _debug_mase_metadata(mg)

if __name__ == "__main__":
    test_emit_verilog_norm()
