from torch import Tensor
import torch.nn as nn
import torch
from typing import Any
import numpy as np

from ....passes.graph.transforms.quantize.quantized_modules.conv2d import (
    Conv2dBinaryResidualSign,
)
from ....passes.graph.transforms.quantize.quantized_modules.linear import (
    LinearBinaryResidualSign,
)

"""
This is the CNV model from the LUTNet Paper

Reference:
https://arxiv.org/pdf/1904.00938.pdf
"""


class CNV(nn.Module):
    def __init__(self, num_classes):
        super(CNV, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding="valid"),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding="valid"),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding="valid"),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding="valid"),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding="valid"),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding="valid"),  # LUT
            nn.BatchNorm2d(256),
        )
        self.seq_blocks_1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
            nn.BatchNorm1d(num_classes),
            # nn.Softmax(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.seq_blocks(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.seq_blocks_1(x)


first_layer_config = {
    "bypass": False,
    "data_in_levels": 2,
    "data_in_residual_sign": False,
    "binary_training": False,
    "data_in_stochastic": False,
    "weight_stochastic": False,
    "data_in_bipolar": True,
    "weight_bipolar": True,
}
hidden_layer_config = {
    "bypass": False,
    "data_in_levels": 2,
    "data_in_residual_sign": True,
    "binary_training": False,
    "data_in_stochastic": False,
    "weight_stochastic": False,
    "data_in_bipolar": True,
    "weight_bipolar": True,
}


class CNV_Residual(nn.Module):
    def __init__(self, num_classes):
        super(CNV_Residual, self).__init__()
        self.seq_blocks = nn.Sequential(
            Conv2dBinaryResidualSign(
                3,
                64,
                3,
                bias=False,
                stride=1,
                padding="valid",
                config=first_layer_config,
            ),
            nn.BatchNorm2d(64),
            Conv2dBinaryResidualSign(
                64,
                64,
                3,
                stride=1,
                bias=False,
                padding="valid",
                config=hidden_layer_config,
            ),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            Conv2dBinaryResidualSign(
                64,
                128,
                3,
                stride=1,
                bias=False,
                padding="valid",
                config=hidden_layer_config,
            ),
            nn.BatchNorm2d(128),
            Conv2dBinaryResidualSign(
                128,
                128,
                3,
                stride=1,
                bias=False,
                padding="valid",
                config=hidden_layer_config,
            ),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            Conv2dBinaryResidualSign(
                128,
                256,
                3,
                stride=1,
                bias=False,
                padding="valid",
                config=hidden_layer_config,
            ),
            nn.BatchNorm2d(256),
            Conv2dBinaryResidualSign(
                256,
                256,
                3,
                stride=1,
                bias=False,
                padding="valid",
                config=hidden_layer_config,
            ),  # LUT
            nn.BatchNorm2d(256),
        )
        self.seq_blocks_1 = nn.Sequential(
            LinearBinaryResidualSign(256, 512, bias=False, config=hidden_layer_config),
            nn.BatchNorm1d(512),
            LinearBinaryResidualSign(512, 512, bias=False, config=hidden_layer_config),
            nn.BatchNorm1d(512),
            LinearBinaryResidualSign(
                512, num_classes, bias=False, config=hidden_layer_config
            ),
            nn.BatchNorm1d(num_classes),
            # nn.Softmax(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.seq_blocks(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.seq_blocks_1(x)


# Getters ------------------------------------------------------------------------------
def get_cnv(
    info,
    pretrained=False,
    **kwargs: Any,
):
    # image_size = info["image_size"]
    num_classes = info.num_classes
    return CNV(num_classes)


def get_cnv_residual(
    info,
    pretrained=False,
    **kwargs: Any,
):
    # image_size = info["image_size"]
    num_classes = info.num_classes
    return CNV_Residual(num_classes)
