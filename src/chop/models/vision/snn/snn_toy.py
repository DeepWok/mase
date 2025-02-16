import torch.nn as nn
from chop.nn.snn import functional
from chop.nn.snn import modules as snn_modules
from chop.nn.snn.modules import neuron as snn_neuron
import torch
from timm.models.registry import register_model
from typing import Any


@register_model
class SNN_toy(nn.Module):
    def __init__(self, tau, num_classes):
        super().__init__()

        self.layer = nn.Sequential(
            snn_modules.Flatten(),
            snn_modules.Linear(28 * 28, num_classes, bias=False),
            snn_neuron.LIFNode(
                tau=tau, surrogate_function=snn_modules.surrogate.ATan()
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


# Getters ------------------------------------------------------------------------------
def get_snn_toy(
    info,
    pretrained=False,
    **kwargs: Any,
):
    tau = info["tau"]
    num_classes = info.num_classes
    return SNN_toy(num_classes)
