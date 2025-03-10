import torch
import torch.nn as nn
from chop.passes.module.module_modify_helper import (
    get_module_by_name,
    set_module_by_name,
)


def replace_by_name_optical(network, module_name: str, new_module):

    original = get_module_by_name(network, module_name)
    updated_module = weight_replacement_optical(original, new_module)
    network = set_module_by_name(network, module_name, updated_module)

    return network


def weight_replacement_optical(original, new_module):
    if isinstance(original, nn.Linear):
        return weight_replacement_linear_optical(original, new_module)
    elif isinstance(original, nn.Conv2d):
        return weight_replacement_conv2d_optical(original, new_module)
    else:
        raise NotImplementedError(
            "weight replacement function for the optical module not implemented"
        )


def weight_replacement_linear_optical(x, y):
    """
    Replace the weights of AllPassMORRCirculantLinear (y)
    with those from a standard nn.Linear (x).
    Focuses only on weight copying (no bias copying).
    """

    # Fetch original linear weight [out_features, in_features]
    W = x.weight.data  # shape: (out_features, in_features)

    # Grab dimensions and zero-pad if needed
    out_features_pad = y.out_features_pad  # padded out_features in y
    in_features_pad = y.in_features_pad  # padded in_features in y
    miniblock = y.miniblock
    grid_dim_y = y.grid_dim_y
    grid_dim_x = y.grid_dim_x

    # Construct padded weight tensor
    W_padded = W.new_zeros((out_features_pad, in_features_pad))
    W_padded[: W.size(0), : W.size(1)] = W  # copy original into top-left

    # Now we create a new tensor of shape [grid_dim_y, grid_dim_x, miniblock]
    # by compressing each row-block [1 x miniblock] from W_padded into a single scalar.
    # This is a simple example that takes the mean across the miniblock slice.
    new_weight = W.new_zeros((grid_dim_y, grid_dim_x, miniblock))

    # Fill new_weight by averaging the corresponding sub-blocks in W_padded
    with torch.no_grad():
        for p in range(grid_dim_y):
            for q in range(grid_dim_x):
                for k in range(miniblock):
                    # The row in W_padded we look at:
                    row_idx = p * miniblock + k
                    # The columns we look at:
                    col_start = q * miniblock
                    col_end = (q + 1) * miniblock

                    block = W_padded[row_idx, col_start:col_end]
                    new_weight[p, q, k] = block.mean()

        # Copy the result into y.weight
        y.load_parameters({"weight": new_weight})
        # y.weight.data.copy_(new_weight)

    return y


def weight_replacement_conv2d_optical(x, y):
    """
    Replace the weights (and bias, if present) of a standard nn.Conv2d (x)
    into an AllPassMORRCirculantConv2d (y).

    Args:
        x (nn.Conv2d):    A standard PyTorch Conv2d module
        y (AllPassMORRCirculantConv2d): An already-constructed optical Conv2d
                                        module into which we copy weights/bias.
    """
    with torch.no_grad():
        # 1) Copy bias (if both x and y actually have one).
        if x.bias is not None and y.bias is not None:
            y.bias.copy_(x.bias)

        # 2) Flatten nn.Conv2d's weight => shape [out_channels, in_channels*kernel_h*kernel_w]
        w_flat = x.weight.data.view(x.out_channels, -1)

        # 3) Zero-pad to match (out_channels_pad, in_channels_pad)
        outC_pad = y.out_channels_pad  # == y.grid_dim_y * y.miniblock
        inC_pad = y.in_channels_pad  # == y.grid_dim_x * y.miniblock

        W = torch.zeros(outC_pad, inC_pad, device=w_flat.device, dtype=w_flat.dtype)
        # Copy as many channels/elements as we have
        W[: x.out_channels, : w_flat.size(1)] = w_flat

        # 4) Reshape into blocks => shape [p, miniblock, q, miniblock]
        p = y.grid_dim_y
        q = y.grid_dim_x
        k = y.miniblock
        W_blocks = W.view(p, k, q, k)  # => [p, k, q, k]

        # 5) For each p,q block, extract the "first column" of size 'k' and place it in y.weight
        #    That is, for a k x k sub-block, we interpret sub_block[:,0] as the "circulant first column".
        for i in range(p):
            for j in range(q):
                sub_block = W_blocks[i, :, j, :]  # shape [k, k]
                y.weight.data[i, j, :] = sub_block[:, 0]

    # Done. At this point, y.weight and y.bias (if present) have been overwritten
    # with a simple block-circulant approximation of x's parameters.
    return y
