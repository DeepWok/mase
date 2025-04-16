from math import ceil, log2

from numpy import ndarray
from torch import Tensor
import torch

from logging import getLogger
from math import ceil
from typing import List

import torch
from torch import Tensor
from torch.autograd.function import InplaceFunction
from torch.nn import functional as F

logger = getLogger(__name__)


# Forced torch gradient overrider
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyFloor(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

my_clamp = MyClamp.apply
my_round = MyRound.apply
my_floor = MyFloor.apply
def _scale_integer_quantize(
    x: Tensor | ndarray, width: int, is_signed: bool = True, quantile: float = 1.0
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if quantile is None:
        quantile = 1.0
    x_max = x.quantile(quantile)
    
    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    
    if is_signed:
        scale = 2**(width - 1) / x_max
    else:
        scale = 2**width / x_max

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale



class ScaleIntegerQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, is_signed: bool = True, quantile: float = 1.0):
        return _scale_integer_quantize(
            x, width=width, is_signed=is_signed, quantile=quantile
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def scale_integer_quantizer(
    x: Tensor | ndarray, width: int, is_signed: bool = True, quantile: float = 1.0
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    return ScaleIntegerQuantize.apply(x, width, is_signed, quantile)
