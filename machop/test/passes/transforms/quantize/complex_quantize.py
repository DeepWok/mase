#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

import toml

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
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.models.toy_custom_fn import ToyCustomFnNet


from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.transforms import (
    quantize_summary_analysis_pass,
    quantize_transform_pass,
)
from chop.passes.utils import deepcopy_mase_graph
from chop.tools.logger import getLogger

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    mlp = ToyCustomFnNet(image_size=(1, 28, 28), num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    mlp(x)
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    # mg = add_mase_ops_analysis_pass(mg, dummy_in)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)

    config_files = [
        "integer.toml",
        "block_fp.toml",
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
        "quantized_ops",
    )

    ori_mg = deepcopy_mase_graph(mg)

    for config_file in config_files:
        # load toml config file
        with open(os.path.join(path, config_file), "r") as f:
            quan_args = toml.load(f)["passes"]["quantize"]
        mg = quantize_transform_pass(mg, quan_args)
        quantize_summary_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
        print(f"Quantize with {config_file} config file successfully!")


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
