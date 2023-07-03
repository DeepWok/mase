#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join("..", "..", "..", "machop"))
from chop.passes.analysis import (add_common_metadata_analysis_pass,
                                  init_metadata_analysis_pass, report,
                                  verify_common_metadata_analysis_pass)
from chop.passes.graph.mase_graph import MaseGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# --------------------------------------------------
#   Model specifications
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
        # w = torch.randn((4, 28 * 28))
        # x = torch.nn.functional.relu(nn.functional.linear(x, w))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def main():
    mlp = MLP()
    mg = MaseGraph(model=mlp)
    print(mg.fx_graph)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)

    # Sanity check and report
    mg = verify_common_metadata_analysis_pass(mg)
    # mg = report(mg)
    # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
