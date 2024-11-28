from chop.nn.snn.modules.linear import Linear
import torch
import torch.nn as nn
from typing import Any, List, Mapping

from chop.nn.snn.modules.spiking_self_attention import DSSA, GWFFN, BN, DownsampleLayer

from chop.nn.snn.modules.conv2d import Conv2d

from chop.nn.snn.modules.pool2d import MaxPool2d, AdaptiveAvgPool2d

from chop.nn.snn.modules.neuron import LIFNode as LIF, ParametricLIFNode as PLIF


class SpikingResformer(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        activation=LIF,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.skip = ["prologue.0", "classifier"]
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode="m"),
                BN(planes[0]),
                MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode="m"),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(
                        planes[idx - 1], planes[idx], stride=2, activation=activation
                    )
                )
                img_size = img_size // 2
            for name in layers[idx]:
                if name == "DSSA":
                    sub_layers.append(
                        DSSA(
                            planes[idx],
                            num_heads[idx],
                            (img_size // patch_sizes[idx]) ** 2,
                            patch_sizes[idx],
                            activation=activation,
                        )
                    )
                elif name == "GWFFN":
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation)
                    )
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.classifier = Linear(planes[-1], num_classes, bias=False, step_mode="m")
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        ret = set()
        for name, module in self.named_modules():
            if isinstance(module, PLIF):
                ret.add(name + ".w")
        return ret
