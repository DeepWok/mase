from math import ceil, log2
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from .utils import block, my_clamp, my_round, unblock


def integer_quantizer(x: Union[Tensor, ndarray], width: int, frac_width: int):
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
    thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), -thresh, thresh - 1) / scale


def integer_fraction(
    width: int, frac_choices: list, min_value: float, max_value: float
):
    max_half_range = max(abs(min_value), abs(max_value))
    int_width = int(log2(max(0.5, max_half_range))) + 2
    frac_width = max(0, width - int_width)
    frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    return frac_width


def minifloat_simple_quantizer(
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
    minifloat_simple_x = (~is_close_to_0)*(sign*(2**exponent)*mantissa) + is_close_to_0*x
    # fmt: on
    return minifloat_simple_x


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


def msfp_quantizer(
    x: Tensor,
    width: int = 12,
    exponent_width: int = 8,
    exponent_bias: int = None,
    block_size: List[int] = [16],
    skip_first_dim: bool = True,
):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    if isinstance(block_size, int):
        block_size = [int]
    # separate x into blocks
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )
    # TODO: Why we have all zero bias
    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()
    # minifloat_simple_quantizer on each block over which a exponent is shared
    mantissa_bits = width - 1
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    mantissa_integer_max = 2**mantissa_bits - 1
    # sign
    per_block_sign = torch.sign(blocked_x + 1e-9)
    # exponent
    per_block_value = torch.abs(blocked_x) + 1e-9
    per_block_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_exponent = my_clamp(per_block_exponent, exponent_min, exponent_max)
    # mantissa
    per_block_mantissa = per_block_value / 2**per_block_exponent
    shift = 2**mantissa_bits
    per_block_mantissa_integer = my_clamp(
        my_round(per_block_mantissa * shift), 0, mantissa_integer_max
    )
    per_block_mantissa = per_block_mantissa_integer / shift

    per_block_msfp = per_block_sign * (2**per_block_exponent) * per_block_mantissa
    msfp_x = unblock(
        per_block_msfp,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    msfp_x = (~is_close_to_0) * msfp_x + (is_close_to_0) * x
    # fmt: on
    return msfp_x


def block_minifloat_quantizer(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: List[int] = 16,
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM), where an exponent bias is shared over all elements in a block
    - `2**-bias_shared x [(-1)^s1 x 2^exponent1 x mantissa1, (-1)^s2 x 2^exponent2 x mantissa2, ...]`
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `width`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_width`: the number of exponent_bits
    - `exponent_bias_width`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    per_block_exponent_bias = my_clamp(
        torch.floor(torch.log2(per_block_max)), 0, 2**exponent_bias_width - 1
    )
    per_block_bm_x = minifloat_ieee_quantizer(
        blocked_x,
        width=width,
        exponent_width=exponent_width,
        exponent_bias=per_block_exponent_bias,
    )

    bm_x = unblock(
        per_block_bm_x,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )
    return bm_x


def block_log_quantizer(
    x: Union[Tensor, ndarray],
    width: int,
    exponent_bias_width: int = None,
    block_size: int = 16,
    skip_first_dim: bool = False,
):
    """
    Convert IEEE FP32/64 to block base-2 log quantized values. A bias is shared over each block

    ---
    - forward: convert IEEE FP32/64 to base-2 log quantized values
    - backward: This is not STE but close to STE because the derivate of (2**exponent) depends on the rounded exponent

    ---
    - `width`: the number of bits, including 1 sign bit and (bits-1) exponent bits
    - `exponent_bias_width`: the number of bits of shared exponent bias
    - `block_size`: a list of integers where each integer is the block size along the corresponding dim

    """
    exponent_bits = width - 1
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    per_block_max_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_bias = my_clamp(
        2**exponent_bits - 1 - per_block_max_exponent, 0, 2**exponent_bias_width - 1
    )

    per_block_lq_x = log_quantizer(blocked_x, width=width, exponent_bias=per_block_bias)
    lq_x = unblock(
        per_block_lq_x,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    return lq_x
