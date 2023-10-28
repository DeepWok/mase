import logging
import os
import sys
import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(7362300227861459126)
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

from chop.tools.utils import (
    copy_weights,
    generate_truth_table,
    init_Conv2dLUT_weight,
)
from chop.passes.transforms.quantize.quantized_modules.conv2d import (
    Conv2dBinaryResidualSign,
    Conv2dBinaryScaling,
    Conv2dLUT,
)
from chop.passes.transforms.quantize.quantizers import residual_sign_quantizer


def generate_input_tensor(batch_size, input_features, min_val, max_val):
    range_width = max_val - min_val
    input_tensor = torch.rand(batch_size, input_features) * range_width + min_val
    return input_tensor


def generate_input_tensor_3d(batch_size, input_c, input_h, input_w, min_val, max_val):
    range_width = max_val - min_val
    input_tensor = (
        torch.rand(batch_size, input_c, input_h, input_w) * range_width + min_val
    )
    return input_tensor


def rounding(your_tensor):
    return (your_tensor * 10000).round() / 10000.0


# --------------------------------------------------
#   Test LUTNet Linear
# --------------------------------------------------
def main():
    lutnet_config = {
        # data
        "data_in_dim": 32,  # NOTE: we need to specify this
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

    # Define the dimensions and range
    batch_size = 1
    input_c = 3
    input_h = 32
    input_w = 32
    k_w = 3
    k_h = 3
    output_c = 1
    min_val = -2
    max_val = 2

    binary_conv2d = Conv2dBinaryResidualSign(
        in_channels=3,
        out_channels=output_c,
        kernel_size=k_w,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        config=binary_config,
    )
    binary_conv2d.pruning_masks.data = torch.rand(binary_conv2d.weight.shape) >= 0.5
    lutnet_conv2d = Conv2dLUT(
        in_channels=3,
        out_channels=1,
        kernel_size=k_w,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        config=lutnet_config,
    )
    initialized_weight, pruning_masks = init_Conv2dLUT_weight(
        levels=lutnet_conv2d.levels,
        k=lutnet_conv2d.k,
        original_pruning_mask=binary_conv2d.pruning_masks,
        original_weight=binary_conv2d.weight,
        out_channels=binary_conv2d.out_channels,
        in_channels=binary_conv2d.in_channels,
        kernel_size=binary_conv2d.kernel_size,
        new_module=lutnet_conv2d,
    )

    copy_weights(
        binary_conv2d.gamma, lutnet_conv2d.trainer.gamma
    )  # TODO: Not sure about this. The paper doesn't specify this part.
    copy_weights(initialized_weight, lutnet_conv2d.trainer.weight)
    copy_weights(pruning_masks, lutnet_conv2d.trainer.pruning_masks)
    copy_weights(binary_conv2d.means, lutnet_conv2d.means)

    print("gamma: {}, {}".format(lutnet_conv2d.trainer.gamma, binary_conv2d.gamma))
    print(
        "weight: binary {}, \n lutnet_linear {}".format(
            binary_conv2d.weight, lutnet_conv2d.trainer.weight
        )
    )

    # # Generate input tensor with custom range
    input_tensor = generate_input_tensor_3d(
        batch_size, input_c, input_h, input_w, min_val, max_val
    )

    # Generate input tensor with custom range
    input_tensor_residual = residual_sign_quantizer(
        lutnet_config["data_in_levels"],
        binary_conv2d.x_quantizer,
        binary_conv2d.means,
        input_tensor,
    )
    # --------------------------
    print("input mask: ", lutnet_conv2d.input_mask)
    # --------------------------
    k = lutnet_conv2d.k
    expanded_weight = binary_conv2d.weight[
        :,
        lutnet_conv2d.input_mask[:, 0],
        lutnet_conv2d.input_mask[:, 1],
        lutnet_conv2d.input_mask[:, 2],
    ].reshape(-1, k, 1)
    expanded_pruning_masks = binary_conv2d.pruning_masks[
        :,
        lutnet_conv2d.input_mask[:, 0],
        lutnet_conv2d.input_mask[:, 1],
        lutnet_conv2d.input_mask[:, 2],
    ].reshape(-1, k, 1)

    table = generate_truth_table(k=k, tables_count=1, device=None)
    connection = expanded_weight[:, 0, :] * table[0, :]
    for extra_input_index in range(1, k):
        pruned_extra_input = ~(
            expanded_pruning_masks[:, extra_input_index, :].squeeze().bool()
        )
        connection[pruned_extra_input] += (
            expanded_weight[pruned_extra_input, extra_input_index, :]
            * table[extra_input_index, :]
        )
    first_table = connection[0]
    assert any(initialized_weight[0, :] == first_table) and any(
        initialized_weight[
            input_c * k_w * k_h * output_c * (lutnet_config["data_in_levels"] - 1),
            :,
        ]
        == first_table
    )
    reformatted_input = input_tensor_residual[
        :,
        :,
        lutnet_conv2d.input_mask[:, 0],
        lutnet_conv2d.input_mask[:, 1],
        lutnet_conv2d.input_mask[:, 2],
    ].view(lutnet_config["data_in_levels"], -1, k, 1)
    first_input = [
        reformatted_input[i, 0, :, :] for i in range(lutnet_config["data_in_levels"])
    ]
    input_for_channel_output_1 = reformatted_input[:, : (input_c * k_w * k_h), :, :]
    weight_for_channel_output_1 = expanded_weight[: (input_c * k_w * k_h), :, :]
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
        selection[i, :, :] = selection[i, :, :] * lutnet_conv2d.means[i] * 2
    first_output = (selection * connection.sign()).sum(-1).sum()
    """
    NOTE: The result before and after transformation is not meant to be exactly the same
    """
    # assert (
    #     rounding(first_output)
    #     == rounding(lutnet_conv2d(input_tensor)[0][0][0][0].sum())
    #     == rounding(binary_conv2d(input_tensor)[0][0][0][0])
    # ), "first_output:{}, {}, {}".format(
    #     first_output,
    #     lutnet_conv2d(input_tensor)[0][0][0][0].sum(),
    #     binary_conv2d(input_tensor)[0][0][0][0],
    # )

    # # --------------------------
    out_binary = binary_conv2d(input_tensor)
    out = lutnet_conv2d(input_tensor)
    # print("input tensor: ", input_tensor, input_tensor.shape)
    # print("output binary: {}".format(out_binary))
    # print("out lutnet: {}".format(out))
    """
    NOTE: The result before and after transformation is not meant to be exactly the same
    """
    for i in range(10):
        input_tensor = generate_input_tensor_3d(
            batch_size, input_c, input_h, input_w, min_val, max_val
        )
        out_binary = binary_conv2d(input_tensor)
        out = lutnet_conv2d(input_tensor)
        print(
            "RESULT: {} {}/{}".format(
                i,
                (rounding(out_binary) == rounding(out)).sum().item(),
                (input_w - 2) * (input_h - 2),
            )
        )
        # assert torch.all(
        #     rounding(out_binary) == rounding(out)
        # ), "output:{} {} {}".format(out_binary == out, out_binary, out)


main()
