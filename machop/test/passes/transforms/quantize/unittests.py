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
    # x = Tensor([0.0000, 0.2152, 3.4068, -3.7986, -0.7431])

    config = {
        "data_in_width": 2,
        "data_in_frac_width": 1,
        "data_out_width": 2,
        "data_out_frac_width": 1,
    }
    logicnets_linear = LinearLogicNets(
        in_features=4, out_features=4, config=config, activation_module="unittest"
    )

    print(logicnets_linear.weight)
    print(logicnets_linear.weight.shape)

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

    lut_out = logicnets_linear(input_tensor)
    print(lut_out)

    logicnets_linear.is_lut_inference = False
    math_out = logicnets_linear(input_tensor)
    print(math_out)

    print(lut_out == math_out)


main()
