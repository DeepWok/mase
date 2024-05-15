#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging
import toml

import torch
import torch.nn as nn

import chop
import chop.passes as passes
from chop.tools.logger import set_logging_verbosity

logger = logging.getLogger("chop")
set_logging_verbosity("info")


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy model of n linear layers
    """

    def __init__(self, num_layers) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(28 * 28, 28 * 28))

        self.layers.append(nn.Linear(28 * 28, 10))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)

        return x


def test_emit_verilog_partition_linear():
    num_layers = 3
    num_devices = 1

    mlp = MLP(num_layers=10)
    mg = chop.MaseGraph(model=mlp)
    print(mg.fx_graph)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg, pass_args={"dummy_in": dummy_in}
    )
    # mg = report_node_shape_analysis_pass(mg)

    # Sanity check and report - verify or compare with expected results here
    # mg, _ = verify_common_metadata_analysis_pass(mg)

    # Quantize to int
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
        "integer.toml",
    )
    mg, _ = passes.report_node_type_analysis_pass(mg)

    # load toml config file
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = passes.quantize_transform_pass(mg, quan_args)

    # There is a bug in the current quantization pass, where the metadata is not uppdated with the precision.
    # Here we temporarily update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            if type(node.meta["mase"].parameters["common"]["args"][arg]) != dict:
                continue
            node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
            node.meta["mase"].parameters["common"]["args"][arg]["precision"] = [8, 3]
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            if type(node.meta["mase"].parameters["common"]["results"][result]) != dict:
                continue
            node.meta["mase"].parameters["common"]["results"][result]["type"] = "fixed"
            node.meta["mase"].parameters["common"]["results"][result]["precision"] = [
                8,
                3,
            ]

    mg, _ = passes.report_node_type_analysis_pass(mg)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg
    )  # add metadata for hardware in each mase node of graph

    partition_args = {
        "cluster_config": None,
        "device_count": num_devices,
        "mode": "naive",
    }
    mg, _ = passes.partition_to_multi_device_transform_pass(mg, partition_args)

    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    # mg, _ = passes.emit_verilog_top_transform_pass(mg)
    # mg, _ = passes.emit_bram_transform_pass(mg)
    # mg, _ = passes.emit_internal_rtl_transform_pass(mg)

    # For internal models, the test inputs can be directly fetched from the dataset
    # using InputGenerator from chop.tools.get_input
    # cosim_config = {"test_inputs": [x], "trans_num": 1}
    # mg, _ = passes.emit_verilog_tb_transform_pass(mg, pass_args=cosim_config)


if __name__ == "__main__":
    test_emit_verilog_partition_linear()
