import torch
import torch.nn as nn
import numpy as np
from chop.nn.optical.modules import optical_module_map
from chop.passes.module.module_modify_helper import get_module_by_name, set_module_by_name

def replace_by_name_optical(
    network,
    module_name: str,
    new_module
):

    original = get_module_by_name(network, module_name)
    updated_module = weight_replacement_optical(original, new_module)
    network = set_module_by_name(network, module_name, updated_module)

    return network


def weight_replacement_optical(x, y):
    """
    Replace the weights of AllPassMORRCirculantLinear (y) 
    with those from a standard nn.Linear (x).
    Focuses only on weight copying (no bias copying).
    """

    # Fetch original linear weight [out_features, in_features]
    W = x.weight.data  # shape: (out_features, in_features)
    
    # Grab dimensions and zero-pad if needed
    out_features_pad = y.out_features_pad   # padded out_features in y
    in_features_pad  = y.in_features_pad    # padded in_features in y
    miniblock        = y.miniblock
    grid_dim_y       = y.grid_dim_y
    grid_dim_x       = y.grid_dim_x
    
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
                    col_end   = (q + 1) * miniblock
                    
                    block = W_padded[row_idx, col_start:col_end]
                    new_weight[p, q, k] = block.mean()
    
        # Copy the result into y.weight
        y.load_parameters({"weight": new_weight})
        # y.weight.data.copy_(new_weight)
    
    return y
