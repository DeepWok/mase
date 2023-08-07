import torch
import torch.nn as nn


class CustomActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + 0.001


# direct alias
custom_activation = CustomActivation.apply


class ToyCustomFnNet(nn.Module):
    def __init__(self, image_size, num_classes, batch_size=1):
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
        self.batch_size = batch_size

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.seq_blocks(x)
        # x = custom_activation(x)
        x = torch.nn.functional.relu(x)
        # testing funcs
        # add
        # This is missing in add_common_metadata.py
        x = x + x
        x = x + 2

        # mult
        # This is missing in add_common_metadata.py
        x = x * x
        x = x * 2

        # sub
        # This is missing in add_common_metadata.py
        x = x - x
        x = x - 2

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
        # x2 = torch.matmul(x.t(), x)

        # WARNING: torch fx graph does not handle this type of run-time definition with data-dependent shapes
        # x = torch.matmul(torch.ones(100, x.size(0)), x)
        x = torch.matmul(
            torch.ones(100, 8), x
        )  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x3072 and 784x100)
        # bmm
        x = x.view(1, x.size(0), x.size(1))
        x = torch.bmm(x, x)
        return x


def get_toyfnnet(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyCustomFnNet(image_size, num_classes)
