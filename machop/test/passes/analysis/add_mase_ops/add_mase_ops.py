#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog

import logging
import os
import sys

import torch

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
from chop.tools.logger import set_logging_verbosity
from chop.passes.graph.mase_graph import MaseGraph
from chop.models.toys.toy_custom_fn import ToyCustomFnNet
from chop.passes.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
)

logger = logging.getLogger("chop.test")
set_logging_verbosity("debug")


def main():
    mlp = ToyCustomFnNet(image_size=(1, 28, 28), num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
