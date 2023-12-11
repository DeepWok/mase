#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)

from chop.actions.train import train
from chop.actions.test import test

from chop.passes.graph.analysis.statistical_profiler import (
    profile_statistics_analysis_pass,
)

from chop.tools.get_input import InputGenerator
from chop.dataset import MaseDataModule

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from chop.passes.graph.utils import deepcopy_mase_graph
from chop.models.utils import MaseModelInfo
from chop.dataset import get_dataset_info
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
class VGG7(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(image_size[0], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512 * 4 * 4, 1024, kernel_size=1),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        x = x.view(-1, 512 * 4 * 4, 1, 1)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = self.last_layer(x)
        return x


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def test_quantize_debug():
    vgg = VGG7([3], 10)
    mg = MaseGraph(model=vgg)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 10
    x = torch.tensor(np.ones((batch_size, 3, 32, 32))).to(torch.float32)
    # x = torch.randn((batch_size, 3, 32, 32))

    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": {"x": x}, "add_value": False}
    )
    mg, _ = add_software_metadata_analysis_pass(mg)

    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    # quant_args = {
    #     "by": "type",
    #     "default": {
    #         "config": {
    #             "name": "ternary",
    #             "data_in_width": 32,
    #             "data_in_scaling_factor": True,
    #             "data_in_mean": None,
    #             "data_in_median": None,
    #             "data_in_max": None,
    #             "weight_scaling_factor": False,
    #             "weight_width": 2,
    #             "weight_mean": None,
    #             "weight_median": None,
    #             "weight_max": None,
    #             "bias_scaling_factor": True,
    #             "bias_width": 32,
    #             "bias_mean": None,
    #             "bias_median": None,
    #             "bias_max": None,
    #         }
    #     },
    #     "linear": {
    #         "config": {
    #             "name": "ternary",
    #             "bypass": True,  # just needed for bypass, other settings are ignored
    #             "data_in_width": 32,
    #             "data_in_scaling_factor": True,
    #             "data_in_mean": None,
    #             "data_in_median": None,
    #             "data_in_max": None,
    #             "weight_scaling_factor": True,
    #             "weight_width": 2,
    #             "weight_mean": None,
    #             "weight_median": None,
    #             "weight_max": None,
    #             "bias_scaling_factor": True,
    #             "bias_width": 32,
    #             "bias_mean": None,
    #             "bias_median": None,
    #             "bias_max": None,
    #         }
    #     },
    # }

    quant_args = {
        "by": "type",
        "default": {
            "config": {
                "name": "binary",
                "data_in_width": 32,
                "data_in_stochastic": False,
                "data_in_bipolar": False,
                "weight_width": 1,
                "weight_stochastic": False,
                "weight_bipolar": False,
                "binary_training": None,
                "bias_width": 32,
                "bias_stochastic": False,
                "bias_bipolar": False,
            }
        },
        "linear": {
            "config": {
                "name": "binary",
                "bypass": True,
                "data_in_width": 32,
                "data_in_stochastic": False,
                "data_in_bipolar": False,
                "weight_width": 1,
                "weight_stochastic": False,
                "weight_bipolar": False,
                "bias_width": 32,
                "bias_stochastic": False,
                "bias_bipolar": False,
            }
        },
    }

    data_module = MaseDataModule(
        model_name="VGG7",
        name="cifar10",
        batch_size=1000,
        num_workers=0,
        tokenizer=None,
        max_token_len=128,
    )
    data_module.prepare_data()
    data_module.setup()

    model_info = MaseModelInfo("vgg7", "vision_others", "vision", True)

    input_generator = InputGenerator(
        model_info=model_info,
        data_module=data_module,
        task="cls",
        which_dataloader="train",
    )

    # stat_args = {
    #     "by": "type",
    #     "target_weight_nodes": ["linear", "conv2d", "maxpool2d", "x"],
    #     "target_activation_nodes": ["relu", "linear", "conv2d", "maxpool2d"],
    #     "activation_statistics": {
    #         "abs_mean": {"dims": "all"},
    #         "range_min_max": {"abs": False, "dims": "all"},
    #         "range_quantile": {"abs": False, "dims": "all", "quantile": 0.5},
    #     },
    #     "weight_statistics": {
    #         "abs_mean": {"dims": "all"},
    #         "range_min_max": {"abs": False, "dims": "all"},
    #         "range_quantile": {"abs": False, "dims": "all", "quantile": 0.5},
    #     },
    #     "input_generator": input_generator,
    #     "num_samples": 32,
    # }

    ori_mg = deepcopy_mase_graph(mg)
    # mg = profile_statistics_analysis_pass(mg, stat_args)
    mg: MaseGraph = quantize_transform_pass(mg, quant_args)

    dataset_info = get_dataset_info("cifar10")
    # plt_args = {"max_epochs": 20, "strategy": "sgd"}
    plt_args = {
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        # "strategy": "ddp_find_unused_parameters_true",
        "strategy": "ddp",
        "precision": "32",
        "callbacks": [],
        "plugins": None,
        "max_epochs": 10,
    }

    # train(
    #     model=mg,
    #     model_info=model_info,
    #     dataset_info=dataset_info,
    #     tokenizer=None,
    #     weight_decay=1e-4,
    #     task="classification",
    #     data_module=data_module,
    #     optimizer="sgd",
    #     learning_rate=1e-1,
    #     plt_trainer_args=plt_args,
    #     auto_requeue=False,
    #     save_path="/home/drv21/mase-tools/mase_output/output",
    #     load_name="/home/drv21/mase-tools/mase_output/vgg7_classification_cifar10_2023-10-06/software/transform/transformed_ckpt/",
    #     load_type="mz",
    # )

    # test(
    #     "vgg7",
    #     info,
    #     mg.model,
    #     "cls",
    #     data_module,
    #     "sgd",
    #     1e-1,
    #     plt_args,
    #     False,
    #     None,
    #     "/home/drv21/mase-tools/mase_output/vgg7_classification_cifar10_2023-09-16/software/training_ckpts/best.ckpt",
    #     "pl",
    # )

    y = mg.model(x)

    # summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
    # mg = report(mg)
    # mg = emit_verilog(mg)


test_quantize_debug()
