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
