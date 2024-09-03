#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging, traceback, pdb
import pytest
import toml

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes

from pathlib import Path

from chop.actions import simulate
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger

set_logging_verbosity("debug")


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)


class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, in_features, hidden_features, out_features) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "linear": {
        "config": {
            "name": "integer_floor",  # quantization scheme name supported are ["integer", "fixed" (equivalent to integer), "lutnet" (dev mode), "logicnets" (dev mode), "binary", "binary_residual", "ternary", "minifloat_ieee", "minifloat_denorm", "log", "block_fp", "block_minifloat", "block_log"]
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 10,
            "weight_frac_width": 3,
            # bias
            "bias_width": 5,
            "bias_frac_width": 2,
            # optional
            "data_out_width": 8,
            "data_out_frac_width": 4,
        },
    },
    "gelu": {
        "config": {
            "name": "integer_floor",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "data_out_width": 8,
            "data_out_frac_width": 4,
        }
    },
}


@pytest.mark.dev
def test_emit_verilog_mlp():
    in_features = 4
    hidden_features = 20
    out_features = 10
    batch_size = 4
    linear = MLP(in_features, hidden_features, out_features)
    mg = chop.MaseGraph(model=linear)
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, in_features))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    # There is a bug in the current quantization pass, where the results metadata is not uppdated with the precision.
    # Here we temporarily update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            if (
                type(node.meta["mase"].parameters["common"]["args"][arg]) == dict
                and "type" in node.meta["mase"].parameters["common"]["args"][arg].keys()
            ):
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            if (
                type(node.meta["mase"].parameters["common"]["results"][result]) == dict
                and "type"
                in node.meta["mase"].parameters["common"]["results"][result].keys()
            ):
                node.meta["mase"].parameters["common"]["results"][result][
                    "type"
                ] = "fixed"
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = [
                    quan_args["linear"]["config"]["data_out_width"],
                    quan_args["linear"]["config"]["data_out_frac_width"],
                ]

    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(skip_build=False, skip_test=False, simulator="verilator", waves=True)


if __name__ == "__main__":
    test_emit_verilog_mlp()