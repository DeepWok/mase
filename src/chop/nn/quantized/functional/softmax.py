from torch import nn
import torch

from chop.nn.quantizers import (
    integer_quantizer,integer_floor_quantizer
)
from math import ceil, log2


def softmax_integer(x: torch.Tensor, dim: int, config: dict, floor=False):
    """
    This function defines the calculation process of hashsoftmax
    Exp result is get from a hash table
    All the data in this function will be quantized to fixed-point
    """
    base_quantizer = integer_floor_quantizer if floor else integer_quantizer
    if config["mult_data"] != None:
        mult = config["mult_data"]
    else:
        mult = 1
    quant_x = base_quantizer(
        x, config["data_in_width"], config["data_in_frac_width"]
    )
    print("quant_x = ",quant_x * 2**config["data_in_frac_width"])
    exp_x = (quant_x*mult).exp()
    quant_exp = base_quantizer(
        exp_x, config["data_in_exp_width"], config["data_in_exp_frac_width"]
    )
    print("quant_exp = ",quant_exp * 2**config["data_in_exp_frac_width"])
    exp_sum = quant_exp.sum(dim=dim, keepdim=True)

    shift_width = config["data_out_frac_width"]
    if torch.all(quant_exp == exp_sum):
        out = torch.tensor(1.0, device=x.device).expand(x.shape)
    else:
        out = quant_exp * (2 ** (shift_width)) // exp_sum
        out = out / (2 ** (shift_width))
    return out
