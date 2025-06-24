#!/usr/bin/env python3
# This example converts a simple MLP model to an ONN model
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


from chop.passes.module.transforms.optical import optical_module_transform_pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test_optical_module_transform_pass():
    model = Net()
    # Sanity check and report
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
        "conv1": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
    }
    optical_module_transform_pass(model, pass_args)


def test_optical_module_transform_pass_2():
    model = Net()
    # Sanity check and report
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "morr_triton",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
        "conv1": {
            "config": {
                "name": "morr_triton",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
    }
    optical_module_transform_pass(model, pass_args)


def test_optical_module_transform_pass_3():
    model = Net()
    pass_args = {
        "by": "regex_name",
        "^fc1$": {
            "config": {"name": "morr_triton", "miniblock": 4},
            "additional": {
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
                "thermal_crosstalk": True,
                "coupling_factor": 0.04,
                "drop_perc": 0.0,
                "phase_noise": True,
                "phase_noise_std": 0.04,
                "in_bit": 8,
                "w_bit": 8,
            },
        },
    }
    new_model, _ = optical_module_transform_pass(model, pass_args)
    print(new_model)


test_optical_module_transform_pass()
test_optical_module_transform_pass_2()
test_optical_module_transform_pass_3()
