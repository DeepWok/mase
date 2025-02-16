import os, sys

from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")

import toml
import torch
import torch.nn as nn
import torch.nn.functional as F

# TO DO: remove
import os

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

import subprocess

# Example command to invoke Verilator
verilator_cmd = ["verilator", "--version"]

# Execute the command and capture output
try:
    output = subprocess.check_output(verilator_cmd, stderr=subprocess.STDOUT, text=True)
    print("Verilator output:", output)
except subprocess.CalledProcessError as e:
    print("Error running Verilator command:", e)

import pytest


class MLP(torch.nn.Module):
    """
    Toy FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = (lambda x: torch.tanh(x))(self.fc1(x))
        return x


@pytest.mark.skip(reason="Not working")
def test_emit_activation_tanh():
    mlp = MLP()
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 2, 2))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    config_file = os.path.join(
        os.path.abspath(""),
        "configs",
        "tests",
        "quantize",
        "fixed.toml",
    )
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = quantize_transform_pass(mg, quan_args)

    _ = report_node_type_analysis_pass(mg)

    # Update the metadata
    for node in mg.fx_graph.nodes:
        for arg, arg_info in node.meta["mase"]["common"]["args"].items():
            if isinstance(arg_info, dict):
                arg_info["type"] = "fixed"
                arg_info["precision"] = [8, 3]
        for result, result_info in node.meta["mase"]["common"]["results"].items():
            if isinstance(result_info, dict):
                result_info["type"] = "fixed"
                result_info["precision"] = [8, 3]

    mg, _ = add_hardware_metadata_analysis_pass(mg, None)

    mg, _ = emit_verilog_top_transform_pass(mg)
    mg, _ = emit_internal_rtl_transform_pass(mg)

    mg, _ = emit_bram_transform_pass(mg)


if __name__ == "__main__":
    test_emit_activation_tanh()
