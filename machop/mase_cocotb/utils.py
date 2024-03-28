import random
from copy import copy

from cocotb.triggers import RisingEdge
import torch
from torch import Tensor
import sys
sys.path.append('../')
from mase_cocotb.z_qlayers import quantize_to_int


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


def large_num_generator(large_num_thres=127, large_num_limit=500, large_num_prob=0.1):
    '''
        Generator large numbers & small numbers with a given probability distribution.
        Default: 500 >= abs(large number) >= 128
    '''
    if (random.random() < large_num_prob):
        if random.random() < 0.5:
            return random.randint(large_num_thres+1, large_num_limit)
        else:
            return random.randint(-large_num_limit, -(large_num_thres+1))
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
    return val # << out_frac_width  # treat data<out_width, out_frac_width> as data<out_width, 0>
