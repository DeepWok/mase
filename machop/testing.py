import os, sys

from chop.ir.graph.mase_graph import MaseGraph
from chop.dataset import MaseDataModule, get_dataset_info

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
    report_node_meta_param_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.get_input import InputGenerator
from chop.models import get_model_info, get_model
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")

import toml
import torch
import torch.nn as nn


# Define simple neural network for digit classification on MNIST dataset.
class MLP(torch.nn.Module):
    """
    Toy FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()
print(data_module.dataset_info)

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=True,
    checkpoint = None)

mg = MaseGraph(model=model)

# Generate MaseGraph and generate meta data.
# mlp = MLP()
# mg = MaseGraph(model=mlp)

# Provide a dummy input for the graph so it can use for tracing
# batch_size = 1
# x = torch.randn((batch_size, 2, 2))
# dummy_in = {"x": x}

# mg, _ = init_metadata_analysis_pass(mg, None)
# mg, _ = add_common_metadata_analysis_pass(
#     mg, {"dummy_in": dummy_in, "add_value": False}
# )


input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)


mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "add_value": False}
)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("common", )})


# ------------------- Quantize model to fixed precision -----------------
config_file = os.path.join(
    os.path.abspath(""),
    "configs",
    "tests",
    "quantize",
    "fixed.toml",
)
with open(config_file, "r") as f:
    quan_args = toml.load(f)["passes"]["quantize"]
mg, _ = quantize_transform_pass(mg, quan_args)

# Ensure the node types are correct after the quantization pass.
_ = report_node_type_analysis_pass(mg)

# Update the metadata
for node in mg.fx_graph.nodes:
    for arg, arg_info in node.meta["mase"]["common"]["args"].items():
        if isinstance(arg_info, dict):
            arg_info["type"] = "fixed"
            arg_info["precision"] = [8, 3]
    for result, result_info in node.meta["mase"]["common"]["results"].items():
        if isinstance(result_info, dict):
            result_info["type"] = "fixed"
            result_info["precision"] = [8, 3]


# ---------------- Run the hardware meta data pass and emit SystemVerilog -----------

from pprint import pprint

def dump_vars(object):
    print("\n")
    pprint(vars(object))

def dump_all(obj):
    print("\n")
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))    


mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("common", "hardware",)})
mg, _ = add_hardware_metadata_analysis_pass(mg, None)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("hardware",)})


mg, _ = emit_verilog_top_transform_pass(mg)
mg, _ = emit_internal_rtl_transform_pass(mg)

# Init block memory.
mg, _ = emit_bram_transform_pass(mg)
# Emit testbench
mg, _ = emit_cocotb_transform_pass(mg)


from chop.actions import simulate

simulate(skip_build=False, skip_test=False)
