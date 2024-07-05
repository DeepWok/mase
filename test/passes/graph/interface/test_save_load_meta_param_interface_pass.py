#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog

import logging
import os
import sys

import torch
from chop.ir.graph import MaseGraph
from chop.models.toys.toy_custom_fn import ToyCustomFnNet

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


from chop.passes.graph.interface import (
    load_node_meta_param_interface_pass,
    save_node_meta_param_interface_pass,
)

from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)

# def test_meta_param_interface_pass():
#     mlp = ToyCustomFnNet(image_size=(1, 28, 28), num_classes=10)
#     mg = MaseGraph(model=mlp)

#     # Provide a dummy input for the graph so it can use for tracing
#     batch_size = 8
#     x = torch.randn((batch_size, 28 * 28))

#     dummy_in = {"x": x}

#     mg, _ = init_metadata_analysis_pass(mg, None)
#     mg, _ = add_common_metadata_analysis_pass(
#         mg, {"dummy_in": dummy_in, "add_value": True})
#     mg, _ = add_software_metadata_analysis_pass(
#         mg, {"dummy_in": dummy_in})
#     mg, _ = save_node_meta_param_interface_pass(mg, "test.toml")
#     mg, _ = load_node_meta_param_interface_pass(mg, "test.toml")

# test_meta_param_interface_pass()
