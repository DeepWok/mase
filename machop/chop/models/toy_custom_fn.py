import torch
import torch.nn as nn


class CustomActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + 0.001


# direct alias
custom_activation = CustomActivation.apply


class ToyCustomFnNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ToyCustomFnNet, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            # linear
            nn.Linear(in_planes, 100),
            # relu
            nn.ReLU(),
        )
        # conv2d
        self.conv2d = nn.Conv2d(1, 100, 3, padding="same")
        # conv1d
        self.conv1d = nn.Conv1d(1, 100, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.seq_blocks(x)
        # x = custom_activation(x)
        x = torch.nn.functional.relu(x)
        # testing funcs
        # add
        # This is missing in add_common_metadata.py
        x = x + x

        # mult
        # This is missing in add_common_metadata.py
        x = x * x

        # sub
        # This is missing in add_common_metadata.py
        x = x - x

        # This is partially supported in add_common_metadata.py
        x = x.view(x.size(0), 1, 10, 10)

        # conv2d
        y = self.conv2d(x)
        # pool
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), x.size(1), -1)
        # conv1d
        x = self.conv1d(x)
        x = x.view(x.size(0), -1)
        # matmul
        # TODO: this fails in add_common_metadata.py
        # x = torch.matmul(x.t, x)
        x = torch.matmul(torch.ones(100, 8), x)
        # bmm
        x = x.view(1, x.size(0), x.size(1))
        x = torch.bmm(x, x)
        return x


def get_toyfnnet(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyCustomFnNet(image_size, num_classes)
