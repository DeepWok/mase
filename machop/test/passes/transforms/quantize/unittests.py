#!/usr/bin/env python3

import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)

from chop.passes.transforms.quantize.quantized_modules.linear import LinearLogicNets


def generate_input_tensor(batch_size, input_features, min_val, max_val):
    range_width = max_val - min_val
    input_tensor = torch.rand(batch_size, input_features) * range_width + min_val
    return input_tensor


# --------------------------------------------------
#   Test LogicNets Linear
# --------------------------------------------------
def main():
    config = {
        "data_in_width": 3,
        "data_in_frac_width": 1,
        "data_out_width": 4,
        "data_out_frac_width": 2,
    }
    logicnets_linear = LinearLogicNets(
        in_features=4, out_features=4, config=config, activation_module="unittest"
    )
    logicnets_linear.calculate_truth_tables()

    # Define the dimensions and range
    batch_size = 10
    input_features = 4
    min_val = -2
    max_val = 2

    # Generate input tensor with custom range
    input_tensor = generate_input_tensor(batch_size, input_features, min_val, max_val)

    print(input_tensor)
    print(input_tensor.shape)

    out = logicnets_linear(input_tensor)
    print(out)

    logicnets_linear.is_lut_inference = False
    out = logicnets_linear(input_tensor)
    print(out)


main()
