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
from utils import (
    update_common_metadata_pass,
    update_hardware_precision_param,
    manually_update_hardware_parallelism_param,
)


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
        # x = self.fc2(x)
        return x


quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "linear": {
        "config": {
            "name": "mxint_hardware",
            # data
            "data_in_width": 4,
            "data_in_exponent_width": 8,
            "data_in_parallelism": [1, 2],
            # weight
            "weight_width": 4,
            "weight_exponent_width": 8,
            "weight_parallelism": [2, 2],
            # bias
            "bias_width": 4,
            "bias_exponent_width": 8,
            "bias_parallelism": [1, 2],
            "data_out_width": 4,
            "data_out_exponent_width": 8,
            "data_out_parallelism": [1, 2],
        }
    },
}


@pytest.mark.dev
def test_emit_verilog_linear():
    in_features = 8
    hidden_features = 32
    out_features = 8
    n = 2
    batch_size = 10
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
    # from chop.passes.graph.transforms.verilog.insert_fork import insert_fifo_after_specified_modules
    # mg, _ = insert_fifo_after_specified_modules(
    #     mg, pass_args = {
    #     "insert_fifo": ["linear"],
    #     "max_parallelism": 2 # used for generating the fifo depth
    #     }
    # )
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    update_hardware_precision_param(mg, quan_args)
    # wp1 = 8
    # wp2 = 1
    # manually_update_hardware_parallelism_param(
    #     mg,
    #     pass_args={
    #         "fc1": {"din": [1, 2], "dout": [1, wp1]},
    #         "fc2": {"din": [1, wp1], "dout": [1, wp2]},
    #     },
    # )
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )


if __name__ == "__main__":
    test_emit_verilog_linear()
    _simulate()
