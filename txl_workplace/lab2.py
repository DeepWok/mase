import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("debug")

batch_size = 8
model_name = "jsc-txl"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# 📝️ change this CHECKPOINT_PATH to the one you trained in Lab1
CHECKPOINT_PATH = "/mnt/c/workplace/projects/ADLS/mase/mase_output/jsc-txl_classification_jsc_2024-01-25/software/training_ckpts/best-v2.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

# get the input generator
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# task 1
# report graph is an analysis pass that shows you the detailed information in the graph
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)
import json
for node in mg.fx_graph.nodes:
    print(node.meta)

# task 2
# pass_args = {
#     "by": "type",                                                            # collect statistics by node name
#     "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
#     "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
#     "weight_statistics": {
#         "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
#     },
#     "activation_statistics": {
#         "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
#     },
#     "input_generator": input_generator,                                      # the input generator for feeding data to the model
#     "num_samples": 32,                                                       # feed 32 samples to the model
# }
#
# mg, _ = profile_statistics_analysis_pass(mg, pass_args)
# mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

# task 3-7
# pass_args = {
#     "by": "type",
#     "default": {"config": {"name": None}},
#     "linear": {
#         "config": {
#             "name": "integer",
#             # data
#             "data_in_width": 8,
#             "data_in_frac_width": 4,
#             # weight
#             "weight_width": 8,
#             "weight_frac_width": 4,
#             # bias
#             "bias_width": 8,
#             "bias_frac_width": 4,
#         }
#     },
# }
#
# from chop.passes.graph.transforms import (
#     quantize_transform_pass,
#     summarize_quantization_analysis_pass,
# )
# from chop.ir.graph.mase_graph import MaseGraph
# from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
# import torch
#
# ori_mg = MaseGraph(model=model)
# ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
# ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})
#
# mg, _ = quantize_transform_pass(mg, pass_args)
# summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
# for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
#     if (type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n))):
#         print(f'difference found at name: {n.name}, mase_type: {get_mase_type(n)}, mase_op: {get_mase_op(n)}\n original module: {type(get_node_actual_target(ori_n))} --> new module {type(get_node_actual_target(n))}')
#         print(f'weight of original module: {get_node_actual_target(ori_n).weight}')
#         print(f'weight of quantized module: {get_node_actual_target(n).weight}')
#         test_input = torch.randn(get_node_actual_target(n).in_features)
#         print(f'random generated test input: {test_input}')
#         print(f'output for original module: {get_node_actual_target(ori_n)(test_input)}')
#         print(f'output for quantized module: {get_node_actual_target(n)(test_input)}')

# task extra
from counter import count_flops_params
# print(dummy_in['x'])
count_flops_params(mg.model, dummy_in['x'], mode='full')
