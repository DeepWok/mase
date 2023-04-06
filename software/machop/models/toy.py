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


def get_toynet(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyNet(image_size, num_classes)


class ToyTiny(nn.Module):
    def __init__(self, image_size, num_classes=1) -> None:
        super().__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.l1 = nn.Linear(in_planes, 2)
        self.relu_1 = nn.ReLU()
        self.l2 = nn.Linear(2, num_classes)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu_1(x)
        x = self.l2(x)
        x = self.relu_2(x)
        return x


def get_toy_tiny(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyTiny(image_size, num_classes)
