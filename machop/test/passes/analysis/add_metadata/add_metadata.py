#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging
import torch
import torch.nn as nn


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)
from chop.passes.graph.mase_graph import MaseGraph

from chop.passes.analysis import (
    add_hardware_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    verify_common_metadata_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_hardware_type_analysis_pass,
)
from chop.passes.transforms import emit_verilog_top_transform_pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------
#   A main function to verify test cases
# --------------------------------------------------
def main():
    mlp = MLP()
    mg = MaseGraph(model=mlp)
    # print(mlp)
    # print(mg.fx_graph)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)
    # mg = report_node_shape_analysis_pass(mg)

    # Sanity check and report - verify or compare with expected results here
    # TODO: Comment it for now - will need to fix verify pass in the future...
    mg = verify_common_metadata_analysis_pass(mg)

    mg = add_hardware_metadata_analysis_pass(mg)
    # mg = report_node_hardware_type_analysis_pass(mg)
    # mg = verify_hardware_metadata_analysis_pass(mg)

    mg = emit_verilog_top_transform_pass(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
