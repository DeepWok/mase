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


class TestModel(nn.Module):
    """Test model with all layors and operations with binarization operation

    A model to test if the quantization for different operations/modules works
    """

    def __init__(self, image_size, num_classes):
        super(TestModel, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=image_size[0],
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 16 * 128 * 128
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 32 * 128 * 128
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),  # 32 * 64 * 64
        )
        self.conv1d = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # 64 * 64 * 64
        self.fc = nn.Linear(64, 100)  # 100
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.seq_blocks(x)
        x = x.view(x.size(0), 32, -1)  # Reshape to [batch_size, 32, height * width]
        x = self.conv1d(x)
        # x = torch.matmul(x, torch.transpose(x, 1, 2))  # Matrix multiplication TODO: add transpose operation to mase
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Add and Subtract operations
        x = x + 10
        x = x - 5
        x = x * 2

        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_testmodel(info, pretrained=False):
    image_size = info["image_size"]
    num_classes = info["num_classes"]
    print(num_classes)
    return TestModel(image_size, num_classes)
