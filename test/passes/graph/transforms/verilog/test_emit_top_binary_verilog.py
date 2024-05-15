# #!/usr/bin/env python3
# # This example converts a simple MLP model to Verilog
# import os, sys, logging
# import toml
#
# import torch
# import torch.nn as nn
#
#
# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         "..",
#         "..",
#         "..",
#         "..",
#         "..",
#         "..",
#         "machop",
#     )
# )
# from chop.ir.graph.mase_graph import MaseGraph
#
# from chop.passes.graph.analysis import (
#     add_hardware_metadata_analysis_pass,
#     add_common_metadata_analysis_pass,
#     init_metadata_analysis_pass,
#     verify_common_metadata_analysis_pass,
#     report_node_type_analysis_pass,
#     report_node_shape_analysis_pass,
#     report_node_hardware_type_analysis_pass,
# )
# from chop.passes.graph.transforms import (
#     quantize_transform_pass,
#     emit_verilog_top_transform_pass,
# )
# from chop.tools.logger import set_logging_verbosity
#
# set_logging_verbosity("debug")
#
#
# # --------------------------------------------------
# #   Model specifications
# #   prefer small models for fast test
# # --------------------------------------------------
# class MLP(torch.nn.Module):
#     """
#     Toy quantized FC model for digit recognition on MNIST
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.fc1 = nn.Linear(28 * 28, 28 * 28)
#         self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
#         # self.fc3 = nn.Linear(28 * 28 * 4, 10)
#
#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1, end_dim=-1)
#         x = torch.nn.functional.relu(self.fc1(x))
#         # x = torch.nn.functional.relu(self.fc2(x))
#         x = self.fc2(x)
#         return x
#
#
# def test_emit_top_binary_verilog():
#     mlp = MLP()
#     mg = MaseGraph(model=mlp)
#     # print(mlp)
#     print(mg.fx_graph)
#
#     # Provide a dummy input for the graph so it can use for tracing
#     batch_size = 1
#     x = torch.randn((batch_size, 28, 28))
#     dummy_in = {"x": x}
#
#     mg = init_metadata_analysis_pass(mg, None)
#     mg = add_common_metadata_analysis_pass(mg, dummy_in)
#     # mg = report_node_shape_analysis_pass(mg)
#
#     # Sanity check and report - verify or compare with expected results here
#     mg = verify_common_metadata_analysis_pass(mg)
#
#     # Quantize to int
#     config_file = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         "..",
#         "..",
#         "..",
#         "..",
#         "configs",
#         "tests",
#         "quantize",
#         "fixed_activation_binary.toml",
#     )
#     mg = report_node_type_analysis_pass(mg)
#
#     # load toml config file
#     with open(config_file, "r") as f:
#         quan_args = toml.load(f)["passes"]["quantize"]
#     mg = quantize_transform_pass(mg, quan_args)
#
#     # There is a bug in the current quantization pass, where the metadata is not uppdated with the precision.
#     # Here we temporarily update the metadata here so we can test the hardware back end.
#
#     # for node in mg.fx_graph.nodes:
#     #     for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
#     #         node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
#     #         node.meta["mase"].parameters["common"]["args"][arg]["precision"] = [8, 3]
#     #     for result, _ in node.meta["mase"].parameters["common"]["results"].items():
#     #         node.meta["mase"].parameters["common"]["results"][result]["type"] = "fixed"
#     #         node.meta["mase"].parameters["common"]["results"][result]["precision"] = [
#     #             8,
#     #             3,
#     #         ]
#
#     mg = report_node_type_analysis_pass(mg)
#     mg = add_hardware_metadata_analysis_pass(mg)
#     mg = report_node_hardware_type_analysis_pass(mg)
#     # mg = verify_hardware_metadata_analysis_pass(mg)
#
#     mg = emit_verilog_top_transform_pass(mg)
