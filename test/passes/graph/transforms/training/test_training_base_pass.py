#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms import (
    training_base_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.logger import set_logging_verbosity

torch.manual_seed(0)
set_logging_verbosity("debug")


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(8 * 8, 8 * 8)
        self.fc2 = nn.Linear(8 * 8, 8 * 8 * 4)
        self.fc3 = nn.Linear(8 * 8 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_S(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(8 * 8, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        return x


def test_training_base():
    mlp = MLP()
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 8, 8))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    quan_args = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "forward": {
                    "pass": "quantize",
                    "name": "integer",
                    "weight_width": 10,
                    "weight_frac_width": 5,
                    "data_in_width": 10,
                    "data_in_frac_width": 5,
                    "bias_width": 10,
                    "bias_frac_width": 5,
                    "data_out_width": 10,
                    "data_out_frac_width": 5,
                },
                "backward": {
                    "pass": "quantize",
                    "name": "integer",
                    "output_grad_width": 10,
                    "output_grad_frac_width": 5,
                    "data_in_width": 10,
                    "data_in_frac_width": 5,
                    "weight_width": 10,
                    "weight_frac_width": 5,
                    "bias_width": 10,
                    "bias_frac_width": 5,
                },
            }
        },
    }

    # deep copy is only possible if we put "add_value" to False
    ori_mg = deepcopy_mase_graph(mg)

    def hook_fn(m, i, o):
        print(m)
        print(
            "------------ Output Grad: Backward Gradient passed by last layer ------------"
        )
        for grad in o:
            try:
                print(grad.shape)
            except AttributeError:
                print("None found for Gradient")
        print("\n")

        print(
            "------------ Input Grad: Gradient computed over Input, Weight, Bias ------------"
        )

        for grad in i:
            try:
                print(grad.shape)
            except AttributeError:
                print("None found for Gradient")

    mlp.fc1.register_backward_hook(hook_fn)
    mlp.fc2.register_backward_hook(hook_fn)
    mlp.fc3.register_backward_hook(hook_fn)
    out = mlp(x)
    (1 - out.mean()).backward()
    print("------------Transformed Model------------")
    mg, _ = training_base_pass(mg, quan_args)
    mg.model.fc1.register_backward_hook(hook_fn)
    mg.model.fc2.register_backward_hook(hook_fn)
    mg.model.fc3.register_backward_hook(hook_fn)
    out = mg.model(x)
    (1 - out.mean()).backward()


def test_training_base_backward_only():
    mlp = MLP_S()
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 8, 8))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    # Sanity check and report
    quan_args_by_pass_forward = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "forward": {
                    "bypass": True,
                    "pass": "quantize",
                    "name": "integer",
                },
                "backward": {
                    "pass": "quantize",
                    "name": "integer",
                    "output_grad_width": 32,
                    "output_grad_frac_width": 24,
                    "data_in_width": 32,
                    "data_in_frac_width": 24,
                    "weight_width": 32,
                    "weight_frac_width": 24,
                    "bias_width": 32,
                    "bias_frac_width": 24,
                },
            }
        },
    }

    # deep copy is only possible if we put "add_value" to False
    ori_mg = deepcopy_mase_graph(mg)

    def hook_fn(m, i, o):
        print(m)
        print(
            "------------ Output Grad: Backward Gradient passed by last layer ------------"
        )
        for grad in o:
            try:
                print(grad.shape)
            except AttributeError:
                print("None found for Gradient")
        print("\n")

        print(
            "------------ Input Grad: Gradient computed over Input, Weight, Bias ------------"
        )

        for grad in i:
            try:
                print(grad.shape)
            except AttributeError:
                print("None found for Gradient")

    mlp.fc1.register_backward_hook(hook_fn)
    out = mlp(x)
    (1 - out.mean()).backward()
    print("------------Transformed Model------------")
    mg, _ = training_base_pass(mg, quan_args_by_pass_forward)
    mg.model.fc1.register_backward_hook(hook_fn)
    out = mg.model(x)
    (1 - out.mean()).backward()


def test_training_base_forward_only():
    mlp = MLP_S()
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 8, 8))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    # Sanity check and report
    quan_args_by_pass_forward = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "forward": {
                    "pass": "quantize",
                    "name": "integer",
                    "weight_width": 10,
                    "weight_frac_width": 5,
                    "data_in_width": 10,
                    "data_in_frac_width": 5,
                    "bias_width": 10,
                    "bias_frac_width": 5,
                    "data_out_width": 10,
                    "data_out_frac_width": 5,
                },
                "backward": {
                    "bypass": True,
                    "pass": "quantize",
                    "name": "integer",
                    "output_grad_width": 32,
                    "output_grad_frac_width": 24,
                    "data_in_width": 32,
                    "data_in_frac_width": 24,
                    "weight_width": 32,
                    "weight_frac_width": 24,
                    "bias_width": 32,
                    "bias_frac_width": 24,
                },
            }
        },
    }

    # deep copy is only possible if we put "add_value" to False
    ori_mg = deepcopy_mase_graph(mg)

    def hook_fn(m, i, o):
        print(m)
        print(
            "------------ Output Grad: Backward Gradient passed by last layer ------------"
        )
        for grad in o:
            try:
                print(grad.shape)
                print(grad)
            except AttributeError:
                print("None found for Gradient")
        print("\n")

        print(
            "------------ Input Grad: Gradient computed over Input, Weight, Bias ------------"
        )

        for grad in i:
            try:
                print(grad.shape)
                print(grad)
            except AttributeError:
                print("None found for Gradient")

    mlp.fc1.register_backward_hook(hook_fn)
    out = mlp(x)
    (1 - out.mean()).backward()
    print("------------Transformed Model------------")
    mg, _ = training_base_pass(mg, quan_args_by_pass_forward)
    mg.model.fc1.register_backward_hook(hook_fn)
    out = mg.model(x)
    (1 - out.mean()).backward()


test_training_base()
test_training_base_backward_only()
test_training_base_forward_only()
