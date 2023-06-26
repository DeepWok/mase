import torch
import torch.nn as nn


class CustomActivation(torch.autograd.Function):

    # WARNING: this dynamic flow can break symbolic trace
    # @staticmethod
    # def forward(ctx, x):
    #     if x > 0:
    #         return x 
    #     else:
    #         return -0.001
    @staticmethod
    def forward(ctx, x):
        return x*x*x-0.001


# direct alias
custom_activation = CustomActivation.apply

class ToyCustomFnNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ToyCustomFnNet, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            nn.Linear(in_planes, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )
        self.final = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.seq_blocks(x.view(x.size(0), -1))
        x = custom_activation(x)
        x = torch.nn.functional.relu(x)
        return self.final(x)


def get_toyfnnet(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return ToyCustomFnNet(image_size, num_classes)

