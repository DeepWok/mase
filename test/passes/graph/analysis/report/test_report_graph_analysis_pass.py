#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog

import logging
import os
import sys

import torch

import chop.models as models
from chop.tools.logger import set_logging_verbosity
from chop.ir.graph import MaseGraph
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report_graph_analysis_pass,
)
from dotdict import DotDict as dd


logger = logging.getLogger("chop.test")
set_logging_verbosity("debug")


def test_report_graph():
    dataset_info = dd()
    dataset_info.num_classes = 10
    dataset_info.image_size = (28, 28, 1)

    mlp = models.get_model("toy", pretrained=False, dataset_info=dataset_info)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    mg, _ = report_graph_analysis_pass(mg, {})


# --------------------------------------------------
#   Execution
# --------------------------------------------------
test_report_graph()
