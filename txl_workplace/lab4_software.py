import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

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

set_logging_verbosity("info")

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
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name

# define a new model
# class JSC_Three_Linear_Layers(nn.Module):
#     def __init__(self):
#         super(JSC_Three_Linear_Layers, self).__init__()
#         self.seq_blocks = nn.Sequential(
#             nn.BatchNorm1d(16),  # 0
#             nn.ReLU(16),  # 1
#             nn.Linear(16, 16),  # linear  2
#             nn.Linear(16, 16),  # linear  3
#             nn.Linear(16, 5),   # linear  4
#             nn.ReLU(5),  # 5
#         )
#
#     def forward(self, x):
#         return self.seq_blocks(x)

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["output_channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["input_channel_multiplier"]
                out_features = out_features * config["output_channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["input_channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "output_channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "input_channel_multiplier": 2,
        "output_channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "input_channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
# mg, _ = redefine_linear_transform_pass(
#     graph=mg, pass_args={"config": pass_config})
#
# for node in mg.fx_graph.nodes:
#     if node.meta["mase"].module is not None:
#         print(node.name, ": ",node.meta["mase"].module)

from torchmetrics.classification import MulticlassAccuracy
import time
from counter import count_flops_params
import matplotlib.pyplot as plt
import copy

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

# build a search space
channel_multipliers = [1,2,3,4,5,6]
search_spaces = []
for cm_config in channel_multipliers:
    pass_config['seq_blocks_2']['input_channel_multiplier'] = cm_config
    pass_config['seq_blocks_2']['output_channel_multiplier'] = cm_config
    pass_config['seq_blocks_4']['input_channel_multiplier'] = cm_config
    pass_config['seq_blocks_4']['output_channel_multiplier'] = cm_config
    pass_config['seq_blocks_6']['input_channel_multiplier'] = cm_config
    pass_config['seq_blocks_6']['output_channel_multiplier'] = cm_config
    # dict.copy() and dict(dict) only perform shallow copies
    # in fact, only primitive data types in python are doing implicit copy when a = b happens
    search_spaces.append(copy.deepcopy(pass_config))

recorded_accs = []
recorded_latencies = []
recorded_model_sizes = []
recorded_flops = []
for i, config in enumerate(search_spaces):
    mg, _ = redefine_linear_transform_pass(
        graph=mg, pass_args={"config": config})
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
        loss = nn.functional.cross_entropy(preds, ys)
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

print('recorded_accs: ', recorded_accs)
print('recorded_latencies: ', recorded_latencies)
print('recorded_model_sizes: ', recorded_model_sizes)
print('recorded_flops: ', recorded_flops)

def plot_bar(x, x_label, x_ticklabels, y_label, width=0.5):
    if len(x) != len(x_ticklabels):
        raise ValueError(f"len(x) != len(x_ticklabels)")
    fig, ax = plt.subplots()
    x_index = list(range(len(x_ticklabels)))

    # plot bar
    rects = ax.bar(x_index, x, width)

    # Adding text for labels
    ax.set_title(f'{x_label} vs {y_label}')
    ax.set_xlabel(x_label)
    ax.set_xticks(x_index)
    ax.set_xticklabels(x_ticklabels)
    ax.set_ylabel(y_label)

    # Adding the actual values on top of the bars for clarity
    for rect in rects:
        height = rect.get_height()
        annotate = height
        if annotate < 10:
            annotate = f'{annotate:.4f}'
        ax.annotate(annotate,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

plot_bar(recorded_accs, 'Channel Multiplier factors', channel_multipliers, 'Accuracies')
plot_bar(recorded_latencies, 'Channel Multiplier factors', channel_multipliers, 'Latencies')
plot_bar(recorded_model_sizes, 'Channel Multiplier factors', channel_multipliers, 'Number of Parameters')
plot_bar(recorded_flops, 'Channel Multiplier factors', channel_multipliers, 'Number of Flops')
plt.show()

