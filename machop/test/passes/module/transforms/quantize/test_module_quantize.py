#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


from chop.passes.module.transforms import quantize_module_transform_pass


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


def test_quantize():
    mlp = MLP()
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    pass_args = {
        "by": "name",
        "fc1": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        },
    }
    quantize_module_transform_pass(mlp, pass_args)

    pass_args = {
        "by": "type",
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "weight_width": 8,
                "weight_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
            }
        },
    }
    quantize_module_transform_pass(mlp, pass_args)
