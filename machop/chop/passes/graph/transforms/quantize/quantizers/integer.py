from math import ceil, log2

from numpy import ndarray
from torch import Tensor
import torch

from .utils import my_clamp, my_round, my_floor


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int = None, is_signed: bool = True
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
    if frac_width is None:
        frac_width = width // 2

    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale


def _integer_floor_quantize(
    x: Tensor, width: int, frac_width: int = None, is_signed: bool = True
):
    if frac_width is None:
        frac_width = width // 2

    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_floor(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_floor(x * scale), int_min, int_max) / scale


class IntegerQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, frac_width: int, is_signed: bool = True):
        return _integer_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class IntegerFloorQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, frac_width: int, is_signed: bool = True):
        return _integer_floor_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def integer_quantizer(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
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
    return IntegerQuantize.apply(x, width, frac_width, is_signed)


def integer_floor_quantizer(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    return IntegerFloorQuantize.apply(x, width, frac_width, is_signed)


def integer_fraction(
    width: int, frac_choices: list, min_value: float, max_value: float
):
    max_half_range = max(abs(min_value), abs(max_value))
    int_width = int(log2(max(0.5, max_half_range))) + 2
    frac_width = max(0, width - int_width)
    frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    return frac_width
