#!/usr/bin/env python3

# This script tests the fixed range augmentation used for the isqrt module.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math


def find_msb_index(x: int, width: int) -> int:
    msb_index = None
    for i in range(1, width+1):
        power = 2 ** (width - i)
        if power <= x:
            msb_index = width - i
            break
    return msb_index

def range_augmentation_sw(x_red: int, msb_index: int, width: int, frac_width: int) -> int:
    ISQRT2 = float_to_int(1 / math.sqrt(2), 1, 15)
    SQRT2 = float_to_int(math.sqrt(2), 1, 15)
    """model of range augmentation for isqrt"""
    shifted_amount = frac_width - msb_index
    shift_amount = None
    res = None

    if shifted_amount > 0:
        if shifted_amount % 2 == 0:
            shift_amount = shifted_amount // 2
            res = x_red
        else:
            shift_amount = (shifted_amount - 1) // 2
            res = x_red * SQRT2 // 2**(width-1)
        res = res * 2 ** (shift_amount)
    elif shifted_amount < 0:
        if shifted_amount % 2 == 0:
            shift_amount = -shifted_amount // 2
            res = x_red
        else:
            shift_amount = (-shifted_amount - 1) // 2
            res = x_red * ISQRT2 // 2**(width-1)
        res = res // 2 ** (shift_amount)
    else:
        res = x_red
    res = res // 2 ** (width - 1 - frac_width)
    return res

def float_to_int(x: float, int_width: int, frac_width: int) -> int:
    integer = int(x)
    x -= integer
    res = integer * (2 ** frac_width)
    for i in range(1, frac_width+1):
        power = 2 ** (-i)
        if power <= x:
            x -= power
            res += 2 ** (frac_width - i)
    return res

def int_to_float(x: int, int_width: int, frac_width: int) -> float:
    integer = x / (2 ** frac_width)
    fraction = x - integer * 2 ** frac_width
    res = integer

    for i in range(1, frac_width+1):
        power = 2 ** (frac_width - i)
        if power < fraction:
            res += 2 ** (-i)
            fraction -= power
    return res

# TODO: this is just for the Q8.0 format. Need to generalise to other formats.
class VerificationCase:
    def __init__(self, samples=1):
        self.int_width = 8
        self.frac_width = 8
        self.data_in_width = self.int_width + self.frac_width
        self.data_in_x = [val for val in range(1, samples+1)]
        self.data_in_msb_index = []
        for x in self.data_in_x:
            msb_index = find_msb_index(x, self.data_in_width)
            self.data_in_msb_index.append(msb_index)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "INT_WIDTH": self.int_width,
            "FRAC_WIDTH": self.frac_width
        }

    def sw_compute(self):
        ref = []
        for x, msb_index in zip(self.data_in_x, self.data_in_msb_index):
            expected = range_augmentation_sw(x, msb_index, self.data_in_width, self.frac_width)
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_range_augmentation(dut):
    """Test for fixed range augmentation for isqrt"""
    samples = 2 ** 16 - 1
    testcase = VerificationCase(samples=samples)

    #for i in range(samples):
    for i in range(samples):
        # Set up module data.
        data_a = testcase.data_in_x[i]
        data_b = testcase.data_in_msb_index[i]

        # Force module data.
        dut.data_a.value = data_a
        dut.data_b.value = data_b
        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Check the output.
        assert (
                dut.data_out.value.integer == expected
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 8, 8)}

            Output:
            Out: {int_to_float(dut.data_out.value.integer, 8, 8)}
            
            Expected: 
            {int_to_float(expected, 8, 8)}
            """

if __name__ == "__main__":
    mase_runner()
