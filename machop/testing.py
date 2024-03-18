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
    summarize_quantization_analysis_pass,
)

from chop.tools.get_input import InputGenerator
from chop.models import get_model_info, get_model
from chop.tools.logger import set_logging_verbosity

from torchmetrics.classification import MulticlassAccuracy

set_logging_verbosity("info")

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

print(model)
mg = MaseGraph(model=model)

# print(mg.modules.named_parameters())
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
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software", )})
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("common", )})

# Save the original mase graph for the sake of comparison with quantised MaseGraph
ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})


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
print(quan_args)
mg, _ = quantize_transform_pass(mg, quan_args)


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

def quantize_and_compare(mg: MaseGraph, ori_mg: MaseGraph):
    pass_args = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": 8,
                "data_in_frac_width": 4,
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
                
                # # stdv
                # "stdv_width": 8,
                # "stdv_frac_width": 4,
                # # mean
                # "mean_width": 8,
                # "mean_frac_width": 4,
            }
        },
    }
    mg, _ = quantize_transform_pass(mg, pass_args)
    summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
    return mg
# mg = quantize_and_compare(mg, ori_mg)



# Ensure the node types are correct after the quantization pass.
_ = report_node_type_analysis_pass(mg)

mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software", )})
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("common", )})

# ------------------------ Own traversal of the original and quantised graphs ------------------- 
from tabulate import tabulate

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.tools.logger import get_logger

logger = get_logger(__name__)


def compared_pre_post_quantized_graphs(
    ori_graph, graph, save_path=None, silent=False
):
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
            "patched_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            actual_target = get_node_actual_target(node)
            if isinstance(actual_target, str):
                return actual_target
            else:
                return actual_target.__name__
        else:
            return node.target

    headers = [
        "Ori name",
        "New name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
    ]
    rows = []
    for ori_n, n in zip(ori_graph.fx_graph.nodes, graph.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
    if not silent:
        logger.debug("Compare nodes:")
        logger.debug("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    logger.info("\n" + tabulate(rows, headers=headers))
    
compared_pre_post_quantized_graphs(ori_mg, mg, save_path=None, silent=False)



# ---------------- Test the software model -----------------------------------------


metric = MulticlassAccuracy(num_classes=5)

num_batches = 5

j = 0
accs = []
for inputs in data_module.train_dataloader():
    xs, ys = inputs

    preds = mg.model(xs)    
    loss = torch.nn.functional.cross_entropy(preds, ys)
    acc = metric(preds, ys)
    accs.append(acc)
    
    print("ACC: ", acc)
    if j > num_batches:
        break
    j += 1

acc_avg = sum(accs) / len(accs)

print("AVG ACC: ", acc_avg)


# ---------------- Run the hardware meta data pass and emit SystemVerilog -----------

from pprint import pprint

def dump_vars(object):
    print("\n")
    pprint(vars(object))

def dump_all(obj):
    print("\n")
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))    


mg, _ = add_hardware_metadata_analysis_pass(mg, None)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("hardware",)})


mg, _ = emit_verilog_top_transform_pass(mg)

mg, _ = emit_bram_transform_pass(mg)

mg, _ = emit_internal_rtl_transform_pass(mg)

# Init block memory.
# Emit testbench
# mg, _ = emit_cocotb_transform_pass(mg)


from chop.actions import simulate

simulate(skip_build=False, skip_test=False)
