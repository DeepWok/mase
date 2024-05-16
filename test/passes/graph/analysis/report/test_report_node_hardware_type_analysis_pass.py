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
        "..",
        "machop",
    )
)

print(sys.path)

from chop.tools.logger import set_logging_verbosity
from chop.ir.graph import MaseGraph
from chop.models.toys.toy import ToyNet
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report_node_hardware_type_analysis_pass,
)

logger = logging.getLogger("chop.test")
set_logging_verbosity("debug")


def test_add_hardware_metadata_analysis_pass():
    mlp = ToyNet(image_size=(1, 28, 28), num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}
    pass_args = {}

    mg, _ = init_metadata_analysis_pass(mg, pass_args)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args)
    mg, _ = add_hardware_metadata_analysis_pass(mg, pass_args)

    mg, _ = report_node_hardware_type_analysis_pass(mg, pass_args)


test_add_hardware_metadata_analysis_pass()
