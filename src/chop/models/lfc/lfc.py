import torch.nn as nn
from typing import Any

"""
This is the LFC model from the LUTNet Paper

Reference:
https://arxiv.org/pdf/1904.00938.pdf
"""


class LFC(nn.Module):
    def __init__(self, image_size, num_classes):
        super(LFC, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            nn.Linear(in_planes, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),  # converted to LUTNet
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.seq_blocks(x.view(x.size(0), -1))


# Getters ------------------------------------------------------------------------------
def get_lfc(
    info,
    pretrained=False,
    **kwargs: Any,
):
    image_size = info["image_size"]
    num_classes = info.num_classes
    return LFC(image_size, num_classes)
