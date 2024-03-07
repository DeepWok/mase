#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math


# AIM: move the the MSB to the leftmost position.
def range_reduction_sw(x: int, width: int) -> int:
    """model of range reduction for isqrt"""
    # Find MSB
    msb_index = None
    for i in range(1, width+1):
        power = 2 ** (width - i)
        if power <= x:
            msb_index = width - i
            break
    res = x
    if msb_index < (width - 1):
        res = res * 2 ** (width - 1 - msb_index)

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
        for x in self.data_in_x:
            expected = range_reduction_sw(x, self.data_in_width) 
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_range_reduction(dut):
    """Test for adding 2 random numbers multiple times"""
    samples = 65536
    testcase = VerificationCase(samples=samples)

    #for i in range(samples):
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
                dut.data_out.value.integer == expected
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 8, 8)}

            Output:
            Out: {int_to_float(dut.data_out.value.integer, 1, 15)}
            
            Expected: 
            {int_to_float(expected, 1, 15)}
            """

if __name__ == "__main__":
    mase_runner()
