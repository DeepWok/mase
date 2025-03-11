from chop.nn.quantized import ViTAttentionInteger

import torch.nn as nn
import torch

from chop.nn.quantized.modules.linear import (
    LinearInteger,
)

if __name__ == "__main__":
    dim = 4
    head = 2

    torch.manual_seed(0)
    x = torch.rand(1, dim, dim)
    q_config = {
        "data_in_width":8,
        "data_in_frac_width":4,
        "qkv_weight_width":8,
        "qkv_weight_frac_width":4,
        "qkv_bias_width":8,
        "qkv_bias_frac_width":4,
        "qkv_width":8,
        "qkv_frac_width":4,
        "qkmm_out_width":4,
        "qkmm_out_frac_width":8,
        "softmax_exp_width":4,
        "softmax_exp_frac_width":8,
        "softmax_out_frac_width":4,
        "svmm_out_width":8,
        "svmm_out_frac_width":4,
    }
    module = ViTAttentionInteger(dim, head, q_config=q_config)
    print(module(x))