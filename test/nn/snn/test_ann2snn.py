import logging
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# import chop

from chop.tools.checkpoint_load import load_model
import numpy as np
import torch
import tqdm
from chop.dataset import MaseDataModule, get_dataset_info
from chop.ir.graph.mase_graph import MaseGraph
from chop import models
from chop.tools.get_input import InputGenerator, get_dummy_input
from chop.actions.train import train
from chop.actions.test import test
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from chop.passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
    verify_common_metadata_analysis_pass,
)

from chop.passes.graph.utils import deepcopy_mase_graph
from chop.passes.graph.transforms.snn.ann2snn import ann2snn_transform_pass

model_name = "cnv-toy"
dataset_name = "cifar10"
BATCH_SIZE = 32

import torch.nn as nn


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(data_loader):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, "reset"):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (
                        (out.argmax(dim=1) == label.to(device)).float().sum().item()
                    )
            total += out.shape[0]
    return correct / total if T is None else corrects / total


# get dataset information
dataset_info = get_dataset_info(dataset_name)

# get model information
model_info = models.get_model_info(model_name)

# get data module
data_module = MaseDataModule(
    model_name=model_name,
    name=dataset_name,
    batch_size=BATCH_SIZE,
    num_workers=8,
    tokenizer=None,
    max_token_len=None,
)
data_module.prepare_data()
data_module.setup()
# NOTE: We only support vision classification models for now.
dummy_input = get_dummy_input(model_info, data_module, "cls", "cpu")

# get an input generator to calibrate the spiking normalization factor during conversion
input_generator = InputGenerator(
    model_info=model_info,
    data_module=data_module,
    task="cls",
    which_dataloader="train",
)

model = models.get_model(model_name, pretrained=False, dataset_info=dataset_info)


# This line transforms a nn.Module to a MaseGraph
mg = MaseGraph(model=model)

# Apply initialization passes to populate information in the graph
mg, _ = init_metadata_analysis_pass(mg, {})
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_input, "add_value": False}
)

# ------------------------------------------------------------
# Training the base ANN
# ------------------------------------------------------------

plt_trainer_args = {
    "max_epochs": 10,
    "devices": 1,
    "accelerator": "cuda",
}

# save_path = "/home/thw20/projects/mase/mase_output/snn/training_ckpts"
# visualizer_save_path = (
#     "/home/thw20/projects/mase/mase_output/snn/software/training_ckpts"
# )
# visualizer = TensorBoardLogger(
#     save_dir=visualizer_save_path,
# )

# train(
#     model=mg.model,
#     model_info=model_info,
#     dataset_info=dataset_info,
#     weight_decay=1e-4,
#     task="cls",
#     data_module=data_module,
#     optimizer="adam",
#     learning_rate=1e-5,
#     plt_trainer_args=plt_trainer_args,
#     scheduler_args=None,
#     save_path=save_path,
#     load_name=None,
#     load_type="pl",
#     visualizer=visualizer,
#     auto_requeue=False,
# )


# train(
#     model=mg.model,
#     model_info=model_info,
#     dataset_info=dataset_info,
#     weight_decay=1e-4,
#     task="cls",
#     data_module=data_module,
#     optimizer="adam",
#     learning_rate=1e-5,
#     plt_trainer_args=plt_trainer_args,
#     scheduler_args=None,
#     save_path=None,
#     load_name=None,
#     load_type="pl",
#     visualizer=None,
#     auto_requeue=False,
# )

# test(
#     model=mg.model,
#     model_info=model_info,
#     data_module=data_module,
#     dataset_info=dataset_info,
#     task="cls",
#     optimizer="adam",
#     learning_rate=1e-5,
#     weight_decay=1e-4,
#     plt_trainer_args=plt_trainer_args,
#     auto_requeue=False,
#     save_path=save_path,
#     visualizer=visualizer,
#     load_name="/home/thw20/projects/mase/mase_output/snn/training_ckpts/best.ckpt",
#     load_type='pl',
# )
# print(val(mg.model, "cuda", data_module.test_dataloader()))


# ann_model = load_model(
#     load_name="/home/thw20/projects/mase/mase_output/snn/training_ckpts/best.ckpt",
#     load_type="pl",
#     model=model,
# )
# print(val(ann_model, "cuda", data_module.test_dataloader()))

# ------------------------------------------------------------
# Convert the base ANN to SNN and test
# ------------------------------------------------------------

quan_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "fuse": True,
    "relu": {
        "config": {
            "name": "IFNode",
            "mode": "99.9%",
            "momentum": 0.1,
        }
    },
    "train_data_loader": input_generator,
    "device": "cpu",  # "device": "cuda",
}

model.to("cpu")  # model.to("gpu")
mg, _ = ann2snn_transform_pass(mg, quan_args)
# print(val(mg.model, "cuda", data_module.test_dataloader(), T=10))


# ------------------------------------------------------------
# load the SNN mz graph and test
# ------------------------------------------------------------
# snn_model = load_model(
#     load_name="/home/thw20/projects/mase/mase_output/cnv_toy_cls_cifar10_2024-10-23/software/transform/transformed_ckpt/graph_module.mz",
#     load_type="mz",
#     model=model,
# )
# print(val(snn_model, "cuda", data_module.test_dataloader(), T=20))
