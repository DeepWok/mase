# THIS FILE IS OUTDATED. PLEASE REFER TO THE unittest_lutnet_conv2d.py
import logging
import os
import sys
import numpy as np

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

from chop.tools.utils import copy_weights, init_LinearLUT_weight, generate_truth_table
from chop.passes.transforms.quantize.quantized_modules.linear import (
    LinearLUT,
    LinearBinaryResidualSign,
)
from chop.passes.transforms.quantize.quantizers import residual_sign_quantizer


def generate_input_tensor(batch_size, input_features, min_val, max_val):
    range_width = max_val - min_val
    input_tensor = torch.rand(batch_size, input_features) * range_width + min_val
    return input_tensor


# --------------------------------------------------
#   Test LUTNet Linear
# --------------------------------------------------
def main():
    lutnet_config = {
        # data
        "data_in_levels": 2,
        "data_in_k": 2,
        "data_in_input_expanded": True,
        "data_in_binarization_level": 1,
        "data_in_width": 1,
        "data_in_frac_width": 0,
        # weight
        "weight_width": 1,
        "weight_frac_width": 0,
        # bias
        "bias_width": 1,
        "bias_frac_width": 0,
    }

    binary_config = {
        "binary_training": True,
        "data_in_levels": 2,
        "data_in_width": 1,
        "data_in_frac_width": 0,
        "data_in_stochastic": False,
        "data_in_bipolar": True,
        "weight_width": 1,
        "weight_stochastic": False,
        "weight_bipolar": True,
        "bias_width": 1,
        "bias_stochastic": False,
        "bias_bipolar": True,
    }

    in_features = 4
    out_features = 4
    binary_linear = LinearBinaryResidualSign(
        config=binary_config,
        in_features=in_features,
        out_features=out_features,
        bias=None,
    )
    binary_linear.pruning_masks.data = torch.rand(binary_linear.weight.shape) >= 0.5
    lutnet_linear = LinearLUT(
        config=lutnet_config,
        in_features=in_features,
        out_features=out_features,
        bias=None,
    )

    initialized_weight, pruning_masks = init_LinearLUT_weight(
        k=lutnet_linear.k,
        levels=binary_linear.levels,
        original_pruning_mask=binary_linear.pruning_masks,
        original_weight=binary_linear.weight,
        in_features=lutnet_linear.in_features,
        out_features=lutnet_linear.out_features,
        new_module=lutnet_linear,
    )
    copy_weights(
        binary_linear.gamma, lutnet_linear.trainer.gamma
    )  # TODO: Not sure about this. The paper doesn't specify this part.
    copy_weights(initialized_weight, lutnet_linear.trainer.weight)
    copy_weights(pruning_masks, lutnet_linear.trainer.pruning_masks)
    copy_weights(binary_linear.means, lutnet_linear.means)

    print("gamma: {}, {}".format(lutnet_linear.trainer.gamma, binary_linear.gamma))
    print(
        "weight: binary {}, \n lutnet_linear {}".format(
            binary_linear.weight, lutnet_linear.trainer.weight
        )
    )

    # Define the dimensions and range
    batch_size = 2
    input_features = 4
    min_val = -2
    max_val = 2

    # Generate input tensor with custom range
    input_tensor = generate_input_tensor(batch_size, input_features, min_val, max_val)
    input_tensor_residual = residual_sign_quantizer(
        lutnet_config["data_in_levels"],
        binary_linear.x_quantizer,
        binary_linear.means,
        input_tensor,
    )
    # --------------------------
    print("input mask: ", lutnet_linear.input_mask)
    # --------------------------
    k = lutnet_linear.k
    mask = lutnet_linear.input_mask.reshape(-1, k * lutnet_linear.in_features)
    bl_expanded_weight = binary_linear.weight[
        np.arange(lutnet_linear.out_features)[:, np.newaxis], mask
    ].reshape(-1, k, 1)
    reformatted_input = input_tensor_residual[:, :, lutnet_linear.input_mask].view(
        lutnet_config["data_in_levels"], -1, k, 1
    )
    first_weight = bl_expanded_weight[0, :, :]
    first_input = [
        reformatted_input[i, 0, :, :] for i in range(lutnet_config["data_in_levels"])
    ]
    first_table = (
        first_weight * generate_truth_table(k=k, tables_count=1, device=None)
    ).sum(dim=-2)
    # assert any(initialized_weight[0, :] == first_table) and any(
    #     initialized_weight[
    #         out_features * in_features * (lutnet_config["data_in_levels"] - 1), :
    #     ]
    #     == first_table
    # )
    input_for_channel_output_1 = reformatted_input[:, : lutnet_linear.in_features, :, :]
    weight_for_channel_output_1 = bl_expanded_weight[: lutnet_linear.in_features, :, :]
    weight_table = (
        weight_for_channel_output_1
        * generate_truth_table(k=k, tables_count=1, device=None)
    ).sum(dim=-2)
    selection = (
        (
            input_for_channel_output_1.sign()
            * generate_truth_table(k=k, tables_count=1, device=None)
            + 1
        )
        / 2
    ).prod(dim=-2)
    # recover_connection_magnitude
    for i in range(lutnet_config["data_in_levels"]):
        selection[i, :, :] = selection[i, :, :] * lutnet_linear.means[i]

    first_output = (selection * weight_table.sign()).sum(-1).sum()
    # assert (
    #     first_output == lutnet_linear(input_tensor)[0][0].sum()
    # ), "first_output:{}, {}".format(
    #     first_output, lutnet_linear(input_tensor)[0][0].sum()
    # )

    # --------------------------
    out_binary = binary_linear(input_tensor)
    out = lutnet_linear(input_tensor)
    # print("input tensor: ", input_tensor, input_tensor.shape)
    # print("output binary: {}".format(out_binary))
    # print("out lutnet: {}".format(out))
    # """
    # NOTE: The result before and after transformation is not meant to be exactly the same
    # """
    # for i in range(10):
    #     input_tensor = generate_input_tensor(
    #         batch_size, input_features, min_val, max_val
    #     )
    #     out_binary = binary_linear(input_tensor)
    #     out = lutnet_linear(input_tensor)
    #     assert torch.all(out_binary == out), "first_output:{}, {}".format(
    #         out_binary, out
    #     )


main()
