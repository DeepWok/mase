#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import toml
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
from chop.models.toy_custom_fn import ToyCustomFnNet
from chop.passes.analysis import (
    add_common_metadata_analysis_pass,
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.passes.utils import deepcopy_mase_graph
from chop.tools.logger import getLogger

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    batch_size = 8
    mlp = ToyCustomFnNet(image_size=(1, 28, 28), num_classes=10, batch_size=batch_size)

    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, 28 * 28))
    mlp(x)

    dummy_in = {"x": x}

    mg = MaseGraph(model=mlp)
    logger.debug(mg.fx_graph)
    mg = init_metadata_analysis_pass(mg, None)
    # mg = add_mase_ops_analysis_pass(mg, dummy_in)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)

    config_files = [
        "integer.toml",
        "block_fp.toml",
        "log.toml",
        "block_log.toml",
        "block_minifloat.toml",
        "binary.toml",
        "minifloat_denorm.toml",
        "minifloat_ieee.toml",
    ]

    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
    )
    ori_mg = deepcopy_mase_graph(mg)

    for config_file in config_files:
        # load toml config file
        with open(os.path.join(path, config_file), "r") as f:
            quan_args = toml.load(f)["passes"]["quantize"]
        mg = quantize_transform_pass(mg, quan_args)
        summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
        print(f"Quantize with {config_file} config file successfully!")


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
