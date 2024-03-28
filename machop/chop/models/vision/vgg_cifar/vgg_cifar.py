# Original code: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/vgg.py
# Paper code https://github.com/Thinklab-SJTU/twns/blob/2c192c38559ffe168139c9e519a053c295ca1313/cls/litenet.py#L86

import torch
import torch.nn as nn


class VGG7(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(image_size[0], 128, kernel_size=3, padding=1),  #0
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   #3
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   #7
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),    #10
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),    #14
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),   #17
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 1024),  #0
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),  #2
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        '''
        x = self.feature_layers[0](x)
        x.retain_grad()
        for i in range(1, 4):
            x = self.feature_layers[i](x)
        x.retain_grad()
        for i in range(4, 8):
            x = self.feature_layers[i](x)
        x.retain_grad()
        for i in range(8, 11):
            x = self.feature_layers[i](x)
        x.retain_grad()       
        for i in range(11, 15):
            x = self.feature_layers[i](x)
        x.retain_grad() 
        for i in range(15, 18):
            x = self.feature_layers[i](x)
        x.retain_grad() 
        '''
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x


def get_vgg7(info, pretrained=False) -> VGG7:
    image_size = info.image_size
    num_classes = info.num_classes
    return VGG7(image_size, num_classes)
