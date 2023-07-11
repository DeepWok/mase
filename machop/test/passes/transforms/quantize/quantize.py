#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

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
from chop.passes.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.transforms import (
    quantize_summary_analysis_pass,
    quantize_transform_pass,
)
from chop.passes.utils import deepcopy_mase_graph
from chop.tools.logger import getLogger

logger = getLogger("chop")
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

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)

    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    quan_args = {
        "by": "type",
        "default": {"config": {"name": None}},
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

    ori_mg = deepcopy_mase_graph(mg)
    mg = quantize_transform_pass(mg, quan_args)

    quantize_summary_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
    # mg = report(mg)
    # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
