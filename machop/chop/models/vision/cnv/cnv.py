from torch import Tensor
import torch.nn as nn
from typing import Any
import numpy as np

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
            # MyFlat(),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq_blocks(x)


# Getters ------------------------------------------------------------------------------
def get_cnv(
    info,
    pretrained=False,
    **kwargs: Any,
):
    # image_size = info["image_size"]
    num_classes = info.num_classes
    return CNV(num_classes)
