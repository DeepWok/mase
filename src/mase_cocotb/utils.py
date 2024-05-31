import random
from copy import copy

from cocotb.triggers import RisingEdge
import torch
from torch import Tensor
import sys

sys.path.append("../")
from mase_cocotb.z_qlayers import quantize_to_int


# Apparently this function only exists in Python 3.12 ...
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def binary_encode(x):
    assert x in [-1, 1]
    return 0 if x == -1 else 1


def binary_decode(x):
    assert x in [0, 1]
    return -1 if x == 0 else 1


async def bit_driver(signal, clk, prob):
    while True:
        await RisingEdge(clk)
        signal.value = 1 if random.random() < prob else 0


def sign_extend_t(value: Tensor, bits: int):
    sign_bit = 1 << (bits - 1)
    return (value.int() & (sign_bit - 1)) - (value.int() & sign_bit)


def sign_extend(value: int, bits: int):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def signed_to_unsigned(value: Tensor, bits: int):
    mask = (1 << bits) - 1
    return value & mask


def floor_rounding(value, in_frac_width, out_frac_width):
    if in_frac_width > out_frac_width:
        return value >> (in_frac_width - out_frac_width)
    elif in_frac_width < out_frac_width:
        return value << (in_frac_width - out_frac_width)
    return value


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def int_floor_quantizer(x: Tensor, width: int, frac_width: int, signed=True):
    if signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    scale = 2**frac_width
    return torch.clamp(torch.floor(x.mul(scale)), int_min, int_max).div(scale)


def random_2d_dimensions():
    compute_dim0 = random.randint(2, 3)
    compute_dim1 = random.randint(2, 3)
    total_dim0 = compute_dim0 * random.randint(1, 3)
    total_dim1 = compute_dim1 * random.randint(1, 3)
    return compute_dim0, compute_dim1, total_dim0, total_dim1


def verilator_str_param(s):
    return f'"{s}"'


def large_num_generator(large_num_thres=127, large_num_limit=500, large_num_prob=0.1):
    """
    Generator large numbers & small numbers with a given probability distribution.
    Default: 500 >= abs(large number) >= 128
    """
    if random.random() < large_num_prob:
        if random.random() < 0.5:
            return random.randint(large_num_thres + 1, large_num_limit)
        else:
            return random.randint(-large_num_limit, -(large_num_thres + 1))
    else:
        return random.randint(-large_num_thres, large_num_thres)


def fixed_cast(val, in_width, in_frac_width, out_width, out_frac_width):
    if in_frac_width > out_frac_width:
        val = val >> (in_frac_width - out_frac_width)
    else:
        val = val << (out_frac_width - in_frac_width)
    in_int_width = in_width - in_frac_width
    out_int_width = out_width - out_frac_width
    if in_int_width > out_int_width:
        if val >> (in_frac_width + out_int_width) > 0:  # positive value overflow
            val = 1 << out_width - 1
        elif val >> (in_frac_width + out_int_width) < -1:  # negative value overflow
            val = -(1 << out_width - 1)
        else:
            val = val
            # val = int(val % (1 << out_width))
    return val  # << out_frac_width  # treat data<out_width, out_frac_width> as data<out_width, 0>


def block_fp_quantize(
    x, width: int = 12, exponent_width: int = 8, exponent: int = None
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
    mantissa_bits = width - 1
    exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    mantissa_integer_max = 2**mantissa_bits - 1
    # sign
    sign = torch.sign(x + 1e-9)
    # exponent
    value = torch.abs(x) + 1e-9
    if exponent == None:
        exponent = torch.ceil(torch.log2(max(x.abs())))
        exponent = torch.clamp(exponent, exponent_min, exponent_max)
    # mantissa
    mantissa = value / 2**exponent
    shift = 2**mantissa_bits
    mantissa_integer = torch.clamp(
        torch.floor(mantissa * shift), 0, mantissa_integer_max
    )
    mantissa = mantissa_integer / shift

    msfp_x = sign * (2**exponent) * mantissa
    return msfp_x, sign * mantissa_integer, exponent
