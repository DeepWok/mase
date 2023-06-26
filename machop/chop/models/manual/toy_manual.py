import torch
import torch.nn as nn
from .base import ManualBase
from chop.passes.transforms.quantizer.layers import LinearInteger

# An example to implement Lienar Integer Quantization using purely CUSTOM OPs
# This is a toy example to show you how to use these customs ops to compose a neural network


class ToyManualNet(ManualBase):
    def __init__(self, image_size, num_classes, config=None):
        super(ToyManualNet, self).__init__(config)

        in_planes = image_size[0] * image_size[1] * image_size[2]

        linear1_config = self.config.get("linear1", None)
        linear2_config = self.config.get("linear2", None)
        linear3_config = self.config.get("linear3", None)
        if any([x is None for x in [linear1_config, linear2_config, linear3_config]]):
            raise ValueError(
                "linear1, linear2, linear3 should not be specified in config"
            )

        self.linear = nn.Sequential(
            LinearInteger(in_planes, 100, config=linear1_config),
            nn.ReLU(),
            LinearInteger(100, 100, config=linear2_config),
            nn.ReLU(),
            LinearInteger(100, num_classes, config=linear3_config),
        )

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


def get_toymanualnet(info, config=None):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyManualNet(image_size, num_classes, config=config)
