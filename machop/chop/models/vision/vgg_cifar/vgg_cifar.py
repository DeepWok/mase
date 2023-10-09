# Original code: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/vgg.py
# Paper code https://github.com/Thinklab-SJTU/twns/blob/2c192c38559ffe168139c9e519a053c295ca1313/cls/litenet.py#L86

import torch
import torch.nn as nn


class VGG7(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(image_size[0], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512 * 4 * 4, 1024, kernel_size=1),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        x = x.view(-1, 512 * 4 * 4, 1, 1)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = self.last_layer(x)
        return x


def get_vgg7(info, pretrained=False) -> VGG7:
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    return VGG7(image_size, num_classes)
