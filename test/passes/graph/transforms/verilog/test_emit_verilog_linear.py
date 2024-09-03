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
from utils import update_common_metadata_pass, update_hardware_precision_param

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
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
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
            "data_out_width": 8,
            "data_out_frac_width": 4,
        },
    },
}


@pytest.mark.dev
def test_emit_verilog_linear():
    in_features = 40
    hidden_features = 20
    out_features = 20
    n = 4
    batch_size = 100
    linear = MLP(in_features, hidden_features, out_features)
    mg = chop.MaseGraph(model=linear)
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, in_features))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    # Increase weight range
    mg.model.fc1.weight = torch.nn.Parameter(
        10 * torch.randn(mg.model.fc1.weight.shape)
    )
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.quantize_transform_pass(mg, quan_args)

    update_common_metadata_pass(mg, quan_args)
    from chop.passes.graph.transforms.verilog.insert_fork import insert_fifo_after_specified_modules
    mg, _ = insert_fifo_after_specified_modules(
        mg, pass_args = {
        "insert_fifo": ["linear"],
        "max_parallelism": 2 # used for generating the fifo depth
        }
    )
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    update_hardware_precision_param(mg, quan_args)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False)


if __name__ == "__main__":
    test_emit_verilog_linear()
