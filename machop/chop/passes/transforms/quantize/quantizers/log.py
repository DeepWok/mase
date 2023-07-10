import torch

from typing import Union
from numpy import ndarray
from torch import Tensor

from .utils import my_clamp, my_round


def log_quantizer(
    x: Union[Tensor, ndarray],
    width: int,
    exponent_bias: Union[int, Tensor, ndarray, None],
):
    """
    - Use non-uniform, base-2 logarithmic representation to encode IEEE FP32/64
    - This quantisation scheme cannot represent 0.

    ---
    - forward: convert IEEE FP32/64 to nearest base-2 log values
    - backward: This is not STE but close to STE because the derivate of (2**exponent) depends on the rounded exponent

    ---
    Currently, base-2 log representation takes the form (-1)**sign_bit * (2**exponent),
    where exponent = intE - exponent_bias, and intE is the unsigned int represented by exponent bits

    ---
    Refer to https://arxiv.org/pdf/1603.01025.pdf
    """

    exponent_bits = width - 1
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_bits - 1) - 1

    exponent_max = 2**exponent_bits - 1 - exponent_bias
    exponent_min = -exponent_bias
    min_pos = 2**exponent_min

    sign = torch.sign(x + min_pos * 0.1)
    value = torch.abs(x) + min_pos * 0.1

    exponent = my_clamp(my_round(torch.log2(value)), exponent_min, exponent_max)

    return sign * (2**exponent)
