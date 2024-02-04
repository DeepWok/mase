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
from chop.passes.graph.transforms import quantize_transform_pass
from chop.passes.graph.transforms.quantize.quantized_modules.linear import LinearInteger
# set_logging_verbosity("debug")
logger = logging.getLogger(__name__)

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
        }
    },
}

def _count_linear_integer(m, x, y, bitops):
    bitops = [8, 64]
    out_features = m.out_features
    if hasattr(m, 'weight_mask'):
        out_features = m.weight_mask.sum() // m.in_features
    total_ops = out_features * m.in_features

    total_bitops = total_ops * bitops[1]

    bias_flops = 1 if m.bias is not None else 0
    total_ops += out_features * bias_flops
    total_bitops += out_features * bias_flops * bitops[0]

    return {
        'flops': total_ops,
        'bitops': total_bitops,
        'params': sum([p.numel() for p in m.parameters()]),
        'weight_shape': tuple(m.weight.size()) if hasattr(m, 'weight') else 0,
    }

def flops_bitops_analysis_pass(graph, pass_args=None):
    if pass_args is None:
        pass_args = {"dummy_in": None, "show_details": False, "quantize_setup": None}
    from counter import count_flops_params

    if pass_args["dummy_in"] is None:
        raise ValueError("Empty dummy_in in pass_args!")
    if not isinstance(graph, MaseGraph):
        raise ValueError("Input graph is not MaseGraph!")

    detail = None
    for node in mg.fx_graph.nodes:
        if node.op == "placeholder":
            if node.name not in pass_args["dummy_in"]:
                raise ValueError("placeholder of graph and dummy_in not match!")
            flops_sum, bitops_sum, _, detail = count_flops_params(graph.model, pass_args["dummy_in"][node.name], custom_ops={LinearInteger: _count_linear_integer,}, verbose=False, mode='full')
            print(f'FLOPs total: {flops_sum}')
            print(f'BITOPs total: {bitops_sum}')
    if detail is None:
        raise ValueError("Placeholder not found in graph!")

    if pass_args["show_details"]:
        from tabulate import tabulate
        rows = []
        headers = ["Name", "Module Type", "Input Size", "Output Size", "Weight Shape", "Flops", "Bitops", "Params"]
        for layer in detail:
            new_row = [
                layer['name'],
                layer['module_type'],
                layer['input_size'],
                layer['output_size'],
                layer['weight_shape'],
                layer['flops'],
                layer['bitops'],
                layer['params'],
            ]
            rows.append(new_row)

        table_txt = tabulate(rows, headers=headers, tablefmt="grid")
        print(table_txt)

mg, _ = quantize_transform_pass(mg, pass_args)

flops_bitops_analysis_pass(mg, {"dummy_in": dummy_in, "show_details": True})