from math import ceil
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from .utils import my_clamp, my_round


def integer_quantizer(x: Union[Tensor, ndarray], bits: int, fraction_bits: int):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    bits: the bit width of the fixed-point number
    decimal_bits: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    thresh = 2 ** (bits - 1)
    scale = 2**fraction_bits

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), -thresh, thresh - 1) / scale


def minifloat_simple_quantizer(
    x: Tensor, bits: int, exponent_bits: int, exponent_bias: int = None
):
    """
    - Converts IEEE FP32/64 to minifloat without the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has no implicit leading bit)
    - backward: STE

    ---
    bits: the bit width of minifloat
    exponent_bits: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (3/8) = -0.00146484375

    ---
    Tested extreme values: large values to saturate, small values close to zero (precision), and 0
    """
    mantissa_bits = bits - exponent_bits - 1

    # default bias value
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_bits - 1) - 1

    exponent_max = 2**exponent_bits - 1 - exponent_bias
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
    x: Tensor, bits: int, exponent_bits: int, exponent_bias: int = None
):
    """
    - Converts IEEE FP32/64 to minifloat with the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has an implicit leading bit)
    - backward: STE

    ---
    bits: the bit width of minifloat
    exponent_bits: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (1+3/8) = -0.00537109375

    ---

    Tested extreme cases: large values to saturate, small normal values, small subnormal values, normal precision, subnormal precision, and 0
    """
    mantissa_bits = bits - exponent_bits - 1

    # set default bias
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_bits - 1) - 1
    # upper and lower bound of shifted exponent
    exponent_max = 2**exponent_bits - 1 - exponent_bias
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
    bits: int,
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

    exponent_bits = bits - 1
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_bits - 1) - 1

    exponent_max = 2**exponent_bits - 1 - exponent_bias
    exponent_min = -exponent_bias
    min_pos = 2**exponent_min

    sign = torch.sign(x + min_pos * 0.1)
    value = torch.abs(x) + min_pos * 0.1

    exponent = my_clamp(my_round(torch.log2(value)), exponent_min, exponent_max)

    return sign * (2**exponent)


def _infer_block_shape(x_shape: List[int], block_shape: List[int]):
    x_ndim = len(x_shape)
    block_ndim = len(block_shape)

    if block_ndim >= x_ndim:
        inferred_block_shape = block_shape[-x_ndim:]
    else:
        inferred_block_shape = [-1] * (x_ndim - block_ndim) + block_shape
    for dim_i in range(x_ndim):
        if (
            inferred_block_shape[dim_i] == -1
            or inferred_block_shape[dim_i] > x_shape[dim_i]
        ):
            inferred_block_shape[dim_i] = x_shape[dim_i]
        else:
            inferred_block_shape[dim_i] = inferred_block_shape[dim_i]
    return inferred_block_shape


def _infer_padded_shape(x_shape: List[int], block_shape: List[int]):
    pad_diff = []
    for x_shape_dim_i, block_shape_dim_i in zip(x_shape, block_shape):
        if block_shape_dim_i == -1 or x_shape_dim_i < block_shape_dim_i:
            pad_diff += [0, 0]
        else:
            num_brackets_dim_i = ceil(x_shape_dim_i / block_shape_dim_i)
            new_x_dim_i = num_brackets_dim_i * block_shape_dim_i
            pad_diff += [new_x_dim_i - x_shape_dim_i, 0]
    pad_diff = pad_diff[::-1]
    return pad_diff


def block(x: Tensor, block_shape: List[int]):
    """
    - Separate input of shape `[B, d1, d2, ..., dn-1]` into blocks using block_shape `[B, b1, b2, ..., bn-1]`, n>=2
    - The output shape will be `[B, b1 x b2 x ... x bn-1, L]`, where the number of blocks `L = ceil(d1/b1) x ceil(d2/b2) x ... x ceil(dn-1/bn-1)`
    - `block_shape` should be a list of integers. The given `block_shape` is aligned to the rightmost element of `x.shape`.
    If `(bi > di)` or `(bi == -1)` or `(bi is not specified because len(block_shape) < len(x.shape))`, then update the value of `bi` using `bi = di`.

    ---
    Return: blocked_x, per_block_max, padded_x_shape, block_shape
    """
    x = x.unsqueeze(1)  # a hack to use F.unfold

    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)

    pad_diff = _infer_padded_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)

    if x.ndim == 3:
        # 2D input
        blocked_x = padded_x.reshape(
            x_shape[0], x_shape[1], padded_x.shape[2] // block_shape[2], block_shape[2]
        )
        # (B, 1, num_buckets, bucket_size)
    elif x.ndim >= 4:
        # 3D or higher dimension
        # 3D: (B, N, hidden_size) in Transformer, N is seq len
        # 4D: (B, C, H, W) in Conv2D
        blocked_x = F.unfold(
            padded_x,
            kernel_size=block_shape[2:],
            dilation=1,
            padding=0,
            stride=block_shape[2:],
        )

    else:
        raise RuntimeError(f"Unsupported input x.ndim {x.ndim-1}")

    per_block_max = torch.abs(blocked_x).max(dim=1, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def unblock(blocked_x: Tensor, padded_x_shape: List[int], block_shape: List[int]):
    """
    The reverse of fold. See function `block`
    """
    blocked_x_shape = [i for i in blocked_x.shape]
    if len(padded_x_shape) == 3:
        x = blocked_x.reshape(
            blocked_x_shape[0],
            blocked_x_shape[1],
            blocked_x_shape[2] * blocked_x_shape[3],
        )
    else:
        x = F.fold(
            blocked_x,
            output_size=padded_x_shape[2:],
            kernel_size=block_shape[2:],
            dilation=1,
            padding=0,
            stride=block_shape[2:],
        )

    return x


def msfp_quantizer(
    x: Tensor,
    bits: int = 12,
    exponent_bits: int = 8,
    exponent_bias: int = None,
    block_size: List[int] = [16],
):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `bits`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_bits`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    if isinstance(block_size, int):
        block_size = [int]
    # separate x into blocks
    x_shape = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size
    )
    # fill zeros to avoid log2(0) = -inf
    per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    # minifloat_simple_quantizer on each block over which a exponent is shared
    mantissa_bits = bits - 1
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_bits - 1) - 1

    exponent_max = 2**exponent_bits - 1 - exponent_bias
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
        per_block_msfp, padded_x_shape=padded_x_shape, block_shape=block_shape
    )
    msfp_x = msfp_x.squeeze(1)
    indexes = []
    for i in range(len(x_shape)):
        indexes.append(slice(None, x_shape[i]))
    msfp_x = msfp_x[indexes]
    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    msfp_x = (~is_close_to_0) * msfp_x + (is_close_to_0) * x
    # fmt: on
    return msfp_x


def block_minifloat_quantizer(
    x: Tensor,
    bits: int,
    exponent_bits: int,
    bias_bits: int,
    block_size: List[int] = 16,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM), where an exponent bias is shared over all elements in a block
    - `2**-bias_shared x [(-1)^s1 x 2^exponent1 x mantissa1, (-1)^s2 x 2^exponent2 x mantissa2, ...]`
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `bits`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_bits`: the number of exponent_bits
    - `bias_bits`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    x_shape = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size
    )

    per_block_exponent_bias = my_clamp(
        torch.floor(torch.log2(per_block_max)), 0, 2**bias_bits - 1
    )
    per_block_bm_x = minifloat_ieee_quantizer(
        blocked_x,
        bits=bits,
        exponent_bits=exponent_bits,
        exponent_bias=per_block_exponent_bias,
    )

    bm_x = unblock(
        per_block_bm_x, padded_x_shape=padded_x_shape, block_shape=block_shape
    )
    indexes = []
    for i in range(len(x_shape)):
        indexes.append(slice(None, x_shape[i]))
    bm_x = bm_x[indexes]
    return bm_x


def block_log_quantizer(
    x: Union[Tensor, ndarray],
    bits: int,
    exponent_bias_bits: int = None,
    block_size: int = 16,
):
    """
    Convert IEEE FP32/64 to block base-2 log quantized values. A bias is shared over each block

    ---
    - forward: convert IEEE FP32/64 to base-2 log quantized values
    - backward: This is not STE but close to STE because the derivate of (2**exponent) depends on the rounded exponent

    ---
    - `bits`: the number of bits, including 1 sign bit and (bits-1) exponent bits
    - `bits`: the number of bits of shared exponent bias
    - `block_size`: a list of integers where each integer is the block size along the corresponding dim

    """
    exponent_bits = bits - 1
    x_shape = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size
    )

    per_block_max_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_bias = my_clamp(
        2**exponent_bits - 1 - per_block_max_exponent, 0, 2**exponent_bias_bits - 1
    )

    per_block_lq_x = log_quantizer(blocked_x, bits=bits, exponent_bias=per_block_bias)
    lq_x = unblock(per_block_lq_x, padded_x_shape, block_shape)

    indexes = []
    for i in range(len(x_shape)):
        indexes.append(slice(None, x_shape[i]))
    lq_x = lq_x[indexes]

    return lq_x
