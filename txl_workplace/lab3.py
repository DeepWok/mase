import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

import matplotlib.pyplot as plt

# figure out the correct path
machop_path = Path("").resolve().parent.parent / "machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model




logger = get_logger("chop")
logger.setLevel(logging.INFO)

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

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

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
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

# grid search
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
import torch
from torchmetrics.classification import MulticlassAccuracy

import time
from counter import count_flops_params
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs = []
recorded_latencies = []
recorded_model_sizes = []
recorded_flops = []
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    latency, flops, model_size = 0, 0, 0
    flag = True
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start = time.time()
        preds = mg.model(xs)
        end = time.time()
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1

        latency += end - start
        if flag:
            flag = False
            flops, model_size, _ = count_flops_params(mg.model, xs, verbose=False, mode='full')
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    print('accs: ', accs)
    print('losses: ', losses)
    print('--------------- divider -----------------')
    recorded_accs.append(acc_avg)
    recorded_latencies.append(latency)
    recorded_model_sizes.append(model_size)
    recorded_flops.append(flops)

# print(recorded_accs)


import matplotlib.pyplot as plt
import numpy as np

def plot_3D_bar_fig(z, title):
    # Data preparation
    x_pos, y_pos = np.meshgrid(np.arange(4), np.arange(4))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    dx = dy = 0.8  # Width and depth of the bars
    dz = np.array(z).reshape(4,4)  # Heights of the bars

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create bars
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, z, shade=True)

    # Customize the plot
    ax.set_xlabel('Data in Frac Widths')
    ax.set_ylabel('Weights in Frac Widths')
    # ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(elev=30, azim=30)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(data_in_frac_widths)
    ax.set_yticklabels(w_in_frac_widths)

    # Show plot
    plt.show()

plot_3D_bar_fig(recorded_accs, title='Accuracies')
plot_3D_bar_fig(recorded_latencies, title='Latencies')
plot_3D_bar_fig(recorded_model_sizes, title='Model Size')
plot_3D_bar_fig(recorded_flops, title='FLOPs')
