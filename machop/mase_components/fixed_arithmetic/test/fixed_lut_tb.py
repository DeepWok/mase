#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math


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

def make_lut(lut_size, width):
    lut_step = 1 / (lut_size + 1)
    x = 1 + lut_step
    lut = []
    for i in range(lut_size):
        value = 1 / math.sqrt(x)
        value = float_to_int(value, 1, width - 1)
        lut.append(value)
        x += lut_step

    for i in range(lut_size):
        print(lut[i])
    return lut

def lut_value_sw(lut, lut_index: int, lut_size: int) -> int:
    return lut[lut_index%lut_size]


# TODO: this is just for the Q8.0 format. Need to generalise to other formats.
class VerificationCase:
    def __init__(self, samples=1):
        self.int_width = 8
        self.frac_width = 8
        self.data_in_width = self.int_width + self.frac_width
        self.data_in_x = [i for i in range(samples)]
        self.samples = samples
        self.lut_pow = 5
        self.lut_size = 2 ** self.lut_pow
        self.lut = make_lut(self.lut_size, self.data_in_width)
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "INT_WIDTH": self.int_width,
            "FRAC_WIDTH": self.frac_width
        }

    def sw_compute(self):
        ref = []
        for x in self.data_in_x:
            expected = lut_value_sw(self.lut, x, self.lut_size)
            ref.append(expected)
        return ref

@cocotb.test()
async def test_fixed_lut(dut):
    """Test for finding LUT value for ISQRT"""
    samples = 32
    testcase = VerificationCase(samples=samples)

    for i in range(samples):
        # Set up module data.
        data_a = testcase.data_in_x[i]

        # Force module data.
        dut.data_a.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Check the output.
        assert (
                dut.data_out.value.integer - expected < 2
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 16, 0)}

            Output:
            Out: {dut.data_out.value.integer}
            
            Expected: 
            {expected}
            """

if __name__ == "__main__":
    mase_runner()
