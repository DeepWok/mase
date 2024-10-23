from torch import Tensor
import torch.nn as nn
import torch
from typing import Any
import numpy as np


class CNN_Toy(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self,x):
        x = self.network(x)
        return x

# Getters ------------------------------------------------------------------------------
def get_cnn_toy(
    info,
    pretrained=False,
    **kwargs: Any,
):
    # image_size = info["image_size"]
    num_classes = info.num_classes
    return CNN_Toy(num_classes)


def get_cnv_residual(
    info,
    pretrained=False,
    **kwargs: Any,
):
    # image_size = info["image_size"]
    num_classes = info.num_classes
    return CNV_Residual(num_classes)
