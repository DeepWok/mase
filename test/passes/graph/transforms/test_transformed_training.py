#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.tools.logger import set_logging_verbosity
from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    report_graph_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms.training.base import SparseLinear
from chop.passes.graph.transforms.training.base import backward_pass_base

set_logging_verbosity("debug")

# short snippet to test SparseLinear
def test_sparse_linear():
    fc = SparseLinear(28 * 28, 28 * 28)
    x = torch.randn((4, 28, 28))
    x = torch.flatten(x, start_dim=1, end_dim=-1)
    y = fc(x)

    # test backward
    target = torch.randn((4, 784))
    loss = torch.nn.functional.mse_loss(y, target)
    optimizer = torch.optim.SGD(fc.parameters(), lr=0.01)
    loss.backward()

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


def test_transformed_training():
    mlp = MLP()
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

    # Sanity check and report
    args = {
        "by": "type",
        "linear": {
            "config": {
                "name": "l1_norm_sparsity",
                # forward
                "forward_w_sparsity": 0.2,
                "forward_x_sparsity": 0.5,
                # backward 
                "backward_x_sparsity": 4,
                "backward_w_sparsity": 8,
                "backward_grad_y_sparsity": 4,
            }
        },
    }
    mg, _ = backward_pass_base(mg, args)
    mg, _ = report_node_type_analysis_pass(mg, {})
    print(mg.model)

test_transformed_training()
