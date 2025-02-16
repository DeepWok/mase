import torch
from torch import Tensor

from .utils import my_clamp, my_round


def _minifloat_denorm_quantize(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias: int = None,
):
    """
    - Converts IEEE FP32/64 to minifloat without the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has no implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (3/8) = -0.00146484375

    ---
    Tested extreme values: large values to saturate, small values close to zero (precision), and 0
    """
    mantissa_bits = width - exponent_width - 1

    # default bias value
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    # if the mantissa is an integer, the max mantissa value will be (2**mantissa_bits -1)
    shifted_mantissa_max = 2**mantissa_bits - 1
    shifted_mantissa_min = 0

    sign = torch.sign(x + 1e-9)

    value = torch.abs(x)
    # ceiling ensures mantissa in the range of [0, 1)
    exponent = torch.ceil(torch.log2(value + 1e-9))
    exponent = my_clamp(exponent, exponent_min, exponent_max)

    # divide value by clipped exponent. this ensures the simulated minifloat value is correct
    # when x is too large (minifloat will saturate) or too close to 0.
    mantissa = value / 2**exponent
    shift = 2**mantissa_bits
    shifted_mantissa = my_round(mantissa * shift)
    # clip the integer mantissa.
    shifted_mantissa = my_clamp(
        shifted_mantissa, shifted_mantissa_min, shifted_mantissa_max
    )
    mantissa = shifted_mantissa / shift
    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(value, torch.tensor([0.0], dtype=value.dtype, device=value.device))
    minifloat_denorm_x = (~is_close_to_0)*(sign*(2**exponent)*mantissa) + is_close_to_0*x
    # fmt: on
    return minifloat_denorm_x


class MinifloatDenormQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        width: int,
        exponent_width: int,
        exponent_bias: int = None,
    ):
        return _minifloat_denorm_quantize(
            x, width=width, exponent_width=exponent_width, exponent_bias=exponent_bias
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def minifloat_denorm_quantizer(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias: int = None,
):
    """
    - Converts IEEE FP32/64 to minifloat without the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has no implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (3/8) = -0.00146484375

    ---
    Tested extreme values: large values to saturate, small values close to zero (precision), and 0
    """
    return MinifloatDenormQuantize.apply(x, width, exponent_width, exponent_bias)


def _minifloat_ieee_quantize(
    x: Tensor, width: int, exponent_width: int, exponent_bias: int = None
):
    """
    - Converts IEEE FP32/64 to minifloat with the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has an implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (1+3/8) = -0.00537109375

    ---

    Tested extreme cases: large values to saturate, small normal values, small subnormal values, normal precision, subnormal precision, and 0
    """
    mantissa_bits = width - exponent_width - 1

    # set default bias
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1
    # upper and lower bound of shifted exponent
    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    # upper and lower bound of shifted minifloat mantissa
    shift = 2**mantissa_bits
    shifted_mantissa_max = 2**mantissa_bits - 1
    shifted_mantissa_min = 0

    sign = torch.sign(x + 1e-9)

    value = torch.abs(x)
    # clip the exponent before calculating mantissa
    exponent = torch.floor(torch.log2(value + 1e-9))
    exponent = my_clamp(exponent, exponent_min, exponent_max)

    mantissa = value / 2**exponent

    shift = 2**mantissa_bits
    # fmt: off
    # if the clipped exponent is zero, the minifloat is in a subnormal form
    # this `is_normal` also help the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    if isinstance(exponent_bias, (int, float)):
        exponent_bias = torch.tensor([exponent_bias], dtype=exponent.dtype, device=exponent.device)
    is_normal = (~torch.isclose(exponent, -exponent_bias))

    shifted_mantissa = is_normal*my_clamp(my_round(mantissa*shift-shift), shifted_mantissa_min, shifted_mantissa_max) +\
        (~is_normal)*my_clamp(my_round(mantissa*shift/2), shifted_mantissa_min, shifted_mantissa_max)
    mantissa = is_normal*(1.0+shifted_mantissa/shift) + (~is_normal)*(shifted_mantissa/shift*2)
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(value, torch.tensor([0.0], dtype=value.dtype, device=value.device))
    minifloat_ieee_x = (~is_close_to_0)*(sign * (2**exponent) * mantissa) + is_close_to_0*x
    # fmt: on
    return minifloat_ieee_x


class MinifloatIEEEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: Tensor, width: int, exponent_width: int, exponent_bias: int = None
    ):
        return _minifloat_ieee_quantize(
            x, width=width, exponent_width=exponent_width, exponent_bias=exponent_bias
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def minifloat_ieee_quantizer(
    x: Tensor, width: int, exponent_width: int, exponent_bias: int = None
):
    """
    - Converts IEEE FP32/64 to minifloat with the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has an implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (1+3/8) = -0.00537109375

    ---

    Tested extreme cases: large values to saturate, small normal values, small subnormal values, normal precision, subnormal precision, and 0
    """
    return MinifloatIEEEQuantize.apply(x, width, exponent_width, exponent_bias)
