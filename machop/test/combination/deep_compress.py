import sys
import torch
from pathlib import Path
from tqdm import tqdm
from math import ceil

from chop.passes.graph.interface.save_and_load import (
    save_mase_graph_interface_pass,
    save_pruned_train_model,
    save_state_dict_ckpt,
)

# Figure out the correct path for Machop
machop_path = Path(".").resolve() / "machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_mase_op
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity
from chop.models import get_model_info, get_model
from chop.actions import test, train

from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    prune_transform_pass,
    add_pruning_metadata_analysis_pass,
    huffman_encode_pass,
    quantize_transform_pass,
)


set_logging_verbosity("info")


# ----------------------------------------- #
#           DEFINE CONFIGURATION            #
# ----------------------------------------- #
batch_size = 512
model_name = "toy_convnet"
dataset_name = "cifar10"
task = "cls"
project_name = "./toy_convnet-compressed-28-03/"
max_epochs = 5
lr = 0.003
optimizer = "adam"

prune_args = {
    "num_iterations": 5,
    "scope": "global",
    "granularity": "elementwise",
    "method": "l1-norm",
    "sparsity": 0.5,
}

quantize_args = {
    "by": "name",
    "default": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 5,
            # weight
            "weight_width": 8,
            "weight_frac_width": 5,
            # bias
            "bias_width": 8,
            "bias_frac_width": 5,
        }
    },
}


# ----------------------------------------- #
#           Prepare model and data          #
# ----------------------------------------- #
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

dataset_info = get_dataset_info(dataset_name)
model_info = get_model_info(model_name)

model = get_model(
    model_name, task=task, dataset_info=dataset_info, pretrained=False, checkpoint=None
)

train_params = {
    "model": model,
    "model_info": model_info,
    "data_module": data_module,
    "dataset_info": dataset_info,
    "task": task,
    "optimizer": optimizer,
    "learning_rate": lr,
    "weight_decay": 0,
    "plt_trainer_args": {
        "max_epochs": max_epochs,
    },
    "auto_requeue": False,
    "save_path": None,
    "visualizer": None,
    "load_name": None,
    "load_type": None,
}

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

mg = MaseGraph(model)
mg, _ = init_metadata_analysis_pass(mg, dummy_in)
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "force_device_meta": False}
)
mg, _ = add_software_metadata_analysis_pass(mg, None)

base_model_save_path = Path(project_name) / "base_model"
base_model_save_path.mkdir(parents=True, exist_ok=True)

save_state_dict_ckpt(mg.model, save_path=base_model_save_path / "model.pth")
# ---------------------------------------------------- #
#            ITERATIVE PRUNING & TRAINING              #
# ---------------------------------------------------- #
print("Starting iterative pruning and training...")
overall_sparsity = prune_args["sparsity"]
num_iterations = prune_args["num_iterations"]

epochs_per_iteration = ceil(max_epochs / num_iterations)
train_params["plt_trainer_args"]["max_epochs"] = epochs_per_iteration

prune_args = {
    "weight": {
        "scope": prune_args["scope"],
        "granularity": prune_args["granularity"],
        "method": prune_args["method"],
    },
    "activation": {
        "scope": prune_args["scope"],
        "granularity": prune_args["granularity"],
        "method": prune_args["method"],
    },
}

original_w_b = {}

for node in mg.fx_graph.nodes:
    if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
        original_w_b[node.name] = {
            "weight": mg.modules[node.target].weight,
            "bias": mg.modules[node.target].bias,
            "meta_weight": node.meta["mase"].parameters["common"]["args"]["weight"][
                "value"
            ],
            "meta_bias": node.meta["mase"].parameters["common"]["args"]["bias"][
                "value"
            ],
        }

for i in tqdm(range(num_iterations)):
    results = train(**train_params)

    iteration_sparsity = 1 - (1 - overall_sparsity) ** ((i + 1) / num_iterations)

    prune_args["weight"]["sparsity"] = iteration_sparsity
    prune_args["activation"]["sparsity"] = iteration_sparsity

    results = train(**train_params)

    iteration_sparsity = 1 - (1 - overall_sparsity) ** ((i + 1) / num_iterations)

    prune_args["weight"]["sparsity"] = iteration_sparsity
    prune_args["activation"]["sparsity"] = iteration_sparsity

    mg, _ = prune_transform_pass(mg, prune_args)

    # Copy the original weights and biases back to the model
    for node in mg.fx_graph.nodes:
        if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
            with torch.no_grad():
                mg.modules[node.target].weight.copy_(original_w_b[node.name]["weight"])
                mg.modules[node.target].bias.copy_(original_w_b[node.name]["bias"])
                # node.meta["mase"].parameters["common"]["args"]["parametrizations.weight.original"]["value"] = original_w_b[node.name]['meta_weight']
                # node.meta["mase"].parameters["common"]["args"]["bias"]["value"] = original_w_b[node.name]['meta_bias']

    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "force_device_meta": False}
    )
    mg, _ = add_software_metadata_analysis_pass(mg, None)
    mg, _ = add_pruning_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": True}
    )

print("Finished iterative pruning and training...")
print("Testing model after pruning and training:")

test(**train_params)

pruned_model_save_path = Path(project_name) / "pruned_model"
pruned_model_save_path.mkdir(parents=True, exist_ok=True)

save_pruned_train_model(mg, pass_args=pruned_model_save_path)
# ---------------------------------------------------- #
#                   QUANTIZATION                       #
# ---------------------------------------------------- #

print("Starting quantization...")
mg, _ = quantize_transform_pass(mg, quantize_args)

print("Finished quantization...")
print("Testing model after quantization:")

test(**train_params)

quantized_model_save_path = Path(project_name) / "quantized_model"
quantized_model_save_path.mkdir(parents=True, exist_ok=True)

save_pruned_train_model(mg, pass_args=quantized_model_save_path)
# ---------------------------------------------------- #
#                   HUFFMAN ENCODING                   #
# ---------------------------------------------------- #

print("Starting Huffman encoding...")
compressed_model_save_path = Path(project_name) / "compressed_model"
compressed_model_save_path.mkdir(parents=True, exist_ok=True)

mg, _ = huffman_encode_pass(mg, {"save_dir": compressed_model_save_path})
