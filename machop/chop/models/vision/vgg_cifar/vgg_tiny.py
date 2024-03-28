import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Convolution 1
            nn.Conv2d(image_size[0], out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Convolution 2
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution 3
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Fully connected layers (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(8 * (image_size[1] // 4) * (image_size[2] // 4), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 8 * (x.size(2) * x.size(3)))
        x = self.classifier(x)
        return x


def get_vgg_tiny(info, pretrained=False) -> TinyVGG:
    image_size = info.image_size
    num_classes = info.num_classes
    return TinyVGG(image_size, num_classes)
