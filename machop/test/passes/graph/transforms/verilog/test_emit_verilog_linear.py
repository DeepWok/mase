#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging
import toml, math

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes
from chop.tools.utils import execute_cli

from pathlib import Path

from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(5 * 5, 5 * 5)
        self.fc2 = nn.Linear(5 * 5, 5 * 5 * 4)
        self.fc3 = nn.Linear(5 * 5 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def test_emit_verilog_linear():
    mlp = MLP()
    mg = chop.MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 5, 5))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    # mg = report_node_shape_analysis_pass(mg)

    # Sanity check and report - verify or compare with expected results here
    # TO DO: fix verify pass
    # mg, _ = passes.verify_common_metadata_analysis_pass(mg)

    # Temporary: check quantization is effective
    for node in mg.fx_graph.nodes:
        if "fc1" in node.name:
            fc1 = mg.modules[node.target]
            a = fc1.weight.clone().detach()

    # Quantize to int
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
        "fixed.toml",
    )

    # load toml config file
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = passes.quantize_transform_pass(mg, quan_args)

    # Temporary: check quantization is effective
    for node in mg.fx_graph.nodes:
        if "fc1" in node.name:
            fc1 = mg.modules[node.target]
            b = fc1.weight.clone().detach()

    print(a)
    print(b)
    assert int(torch.Tensor.sum(torch.eq(a, b).int())) != int(
        math.prod(a.shape)
    ), "Quantization is not effective!"

    # inspect the graph metadata
    # mg, _ = passes.report_node_meta_param_analysis_pass(mg)

    # add metadata for hardware in each mase node of graph
    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)
    # pretty print
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)
    # mg = verify_hardware_metadata_analysis_pass(mg)

    # Emit Verilog sources
    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)

    # Test Verilog sources
    mg, _ = passes.test_verilog_analysis_pass(mg)


if __name__ == "__main__":
    test_emit_verilog_linear()
