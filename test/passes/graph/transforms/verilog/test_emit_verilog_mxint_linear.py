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
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

parallelism = 32
quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "linear": {
        "config": {
            "name": "mxint_hardware",
            # data
            "data_in_width": 8,
            "data_in_exponent_width": 8,
            "data_in_parallelism": [1, parallelism],
            # weight
            "weight_width": 6,
            "weight_exponent_width": 8,
            "weight_parallelism": [parallelism, parallelism],
            # bias
            "bias_width": 6,
            "bias_exponent_width": 8,
            "bias_parallelism": [1, parallelism],
            "data_out_width": 8,
            "data_out_exponent_width": 8,
            "data_out_parallelism": [1, parallelism],
            "round_bits": 4,
        }
    },
}



from mxint_quant import vit_module_level_quantize, VIT_CUSTOM_OPS
@pytest.mark.dev
def test_emit_verilog_linear():
    in_features = 192
    hidden_features = 192*4
    out_features = 192
    n = 196
    batch_size = 10
    linear = MLP(in_features, hidden_features, out_features)
    qlinear = vit_module_level_quantize(linear, q_config=quan_args)
    qlinear.fc1.weight = torch.nn.Parameter(
        10 * torch.randn(qlinear.fc1.weight.shape) - 5
    )
    model_path = "/scratch/cx922/mase/mlp_model.pth"
    torch.save(qlinear.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    mg = chop.MaseGraph(model=qlinear, custom_ops=VIT_CUSTOM_OPS)
    # Save the whole model to a file
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, in_features))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    # Increase weight range
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    update_common_metadata_pass(mg, quan_args)

    mg, _ = passes.add_hardware_metadata_analysis_pass( mg, pass_args={"max_parallelism": [2] * 4})
    update_hardware_precision_param(mg, quan_args)

    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path("/scratch/cx922/mase/mxint_linear_m8e6"),
    }

    mg, _ = passes.emit_verilog_top_transform_pass(mg, pass_args)
    mg, _ = passes.emit_bram_transform_pass(mg, pass_args)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg, pass_args)
    mg, _ = passes.emit_vivado_project_transform_pass(mg, pass_args)


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )


if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
