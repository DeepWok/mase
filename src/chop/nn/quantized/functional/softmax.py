from torch import nn
import torch

from chop.nn.quantizers import (
    integer_quantizer,
)
from math import ceil, log2


def softmax_integer(x: torch.Tensor, dim: int, config: dict):
    """
    This function defines the exact calculation process of hashsoftmax
    The main modification is that the exp result is get from a hash table but not be calculated
    All the data in this function will be represented with fixed-point representation
    """
    quant_x = integer_quantizer(
        x, config["data_in_width"], config["data_in_frac_width"]
    )
    exp_x = quant_x.exp()
    quant_exp = integer_quantizer(
        exp_x, config["data_in_exp_width"], config["data_in_exp_frac_width"]
    )
    exp_sum = quant_exp.sum(dim=dim, keepdim=True)

    shift_width = config["data_in_div_frac_width"]
    if torch.all(quant_exp == exp_sum):
        out = torch.tensor(1.0, device=x.device).expand(x.shape)
    else:
        out = quant_exp * (2 ** (shift_width)) // exp_sum
        out = out / (2 ** (shift_width))
    return out
