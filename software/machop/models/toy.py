import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ToyNet, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            nn.Linear(in_planes, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.seq_blocks(x.view(x.size(0), -1))


def get_toynet(info):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyNet(image_size, num_classes)
