#!/usr/bin/env python3
# This example to quantize a simple ToyNet model to LUTNet (no pruning, no retraining)
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
# A simple convolutional toy net that uses Conv2d, Conv1d and Linear layers.  This
# network is primarily used to test the pruning transformation.
class ToyConvNet(nn.Module):
    def __init__(self, num_classes, channels=[3, 8, 16, 32, 64]):
        super(ToyConvNet, self).__init__()
        self.channels = channels
        # self.linear = nn.Linear(
        #     32, num_classes
        # )  # enable this line to test linear quantization
        self.block_1 = self._conv_block(nn.Conv2d, channels[0], channels[1], 3, 1, 1)
        self.block_2 = self._conv_block(nn.Conv2d, channels[1], channels[2], 3, 1, 1)
        self.block_3 = self._conv_block(nn.Conv1d, channels[2], channels[3], 3, 1, 1)
        self.block_4 = self._conv_block(nn.Conv1d, channels[3], channels[4], 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(channels[4], num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.maxpool(x)
        x = self.block_2(x)
        x = x.view(x.size(0), self.channels[2], -1)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    # Helper functions -----------------------------------------------------------------
    def _conv_block(self, conv_class, *args):
        return nn.Sequential(conv_class(*args), nn.ReLU())


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def test_quantize_lutnet_linear_3():
    mlp = ToyConvNet(num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 3, 32, 32))
    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)

    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)

    # NOTE: To turn a convolutional layer to LUTNet, we will need a manually pass in the input dimension in the toml file. This is use to obtain the output dimension.
    quan_args = {
        "by": "type",
        "default": {"config": {"name": None}},
        "baseline_weight_path": "/workspace/mase_output/toy_classification_cifar10_2023-08-21/software/transform/transformed_ckpt_bl/transformed_ckpt/graph_module.mz",
        "conv2d": {
            "config": {
                "name": "lutnet",
                # data
                "data_in_k": 2,
                "data_in_input_expanded": True,
                "data_in_binarization_level": "binarized_weight",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_in_dim": 32,
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                "weight_k": "NA",
                "weight_input_expanded": "NA",
                "weight_binarization_level": "NA",
                "weight_in_dim": "NA",
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
                "bias_k": "NA",
                "bias_input_expanded": "NA",
                "bias_binarization_level": "NA",
                "bias_in_dim": "NA",
            }
        },
        "linear": {
            "config": {
                "name": "lutnet",
                # data
                "data_in_k": 2,
                "data_in_input_expanded": True,
                "data_in_binarization_level": "binarized_weight",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_in_dim": "NA",
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                "weight_k": "NA",
                "weight_input_expanded": "NA",
                "weight_binarization_level": "NA",
                "weight_in_dim": "NA",
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
                "bias_k": "NA",
                "bias_input_expanded": "NA",
                "bias_binarization_level": "NA",
                "bias_in_dim": "NA",
            }
        },
    }

    ori_mg = deepcopy_mase_graph(mg)
    # NOTE: A proper baseline checkpoint is needed to run this transform. Specify the baseline checkpoint in baseline_weight_path
    # mg = quantize_transform_pass(mg, quan_args)

    summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
    # mg = report(mg)
    # mg = emit_verilog(mg)
