import os, sys
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
    test_verilog_analysis_pass,
)
from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    quantize_transform_pass,
)
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")

import toml
import torch
import torch.nn as nn

import os


# %%
class MLP(torch.nn.Module):
    """
    Toy FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


mlp = MLP()
mg = MaseGraph(model=mlp)

# Provide a dummy input for the graph so it can use for tracing
batch_size = 1
x = torch.randn((batch_size, 28, 28))
dummy_in = {"x": x}

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "add_value": False}
)
mg, _ = add_hardware_metadata_analysis_pass(mg, None)

config_file = os.path.join(
    os.path.abspath(""),
    "..",
    "..",
    "..",
    "..",
    "..",
    "..",
    "machop",
    "configs",
    "tests",
    "quantize",
    "fixed.toml",
)
with open(config_file, "r") as f:
    quan_args = toml.load(f)["passes"]["quantize"]
mg, _ = quantize_transform_pass(mg, quan_args)

_ = report_node_type_analysis_pass(mg)

# Update the metadata
for node in mg.fx_graph.nodes:
    for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
        if isinstance(arg_info, dict):
            arg_info["type"] = "fixed"
            arg_info["precision"] = [8, 3]
    for result, result_info in (
        node.meta["mase"].parameters["common"]["results"].items()
    ):
        if isinstance(result_info, dict):
            result_info["type"] = "fixed"
            result_info["precision"] = [8, 3]

mg, _ = emit_verilog_top_transform_pass(mg)
mg, _ = emit_internal_rtl_transform_pass(mg)

mg, _ = emit_bram_transform_pass(mg)
mg, _ = test_verilog_analysis_pass(mg)

# TO DO: implement this as an action
# def run_tb_action(mg):
#     SIM = getenv("SIM", "verilator")
#     runner = get_runner(SIM)
#     sources = [
#         Path(mase_components.__file__).parent.joinpath(dep)
#         for dep in mg.fx_graph.meta["mase"].parameters["hardware"]["verilog_sources"]
#     ]
#     sources += ["top/hardware/rtl/top.sv"]
#     runner.build(
#         verilog_sources=sources,
#         includes=["top/hardware/rtl"],
#         hdl_toplevel="top",
#         build_args=[
#             # "--Wall",
#             # Turn on assertions
#             "--assert",
#         ],
#         parameters=[],  # use default parameters,
#         build_dir="build",
#     )
#     runner.test(
#         hdl_toplevel="top",
#         test_module="top_tb",
#     )
#     num_tests, fail = get_results("build/results.xml")
#     return num_tests, fail
#
#
# run_tb_action(mg)
