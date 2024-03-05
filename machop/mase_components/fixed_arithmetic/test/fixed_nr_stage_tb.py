#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math


def nr_stage_sw(x: int, lut: int) -> int:
    """model of newton raphson stage"""
    x = int_to_float(x, 1, 15)
    lut = int_to_float(lut, 1, 15)
    res = lut * (1.5 - 0.5 * x * lut * lut)
    return float_to_int(res, 1, 15)

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
        self.data_in_width = 16
        self.int_width = 1
        self.frac_width = 15
        self.data_in_x = [float_to_int(random.random(), self.int_width, self.frac_width) for _ in range(samples)]
        self.data_in_lut = [float_to_int(random.random(), self.int_width, self.frac_width) for _ in range(samples)]
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
        for x, lut in zip(self.data_in_x, self.data_in_lut):
            expected = nr_stage_sw(x, lut) 
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_nr_stage(dut):
    """Test for adding 2 random numbers multiple times"""
    samples = 100
    testcase = VerificationCase(samples=samples)

    #for i in range(samples):
    for i in range(samples):
        # Set up module data.
        data_a = testcase.data_in_x[i]
        data_b = testcase.data_in_lut[i]

        # Force module data.
        dut.data_a.value = data_a
        dut.data_b.value = data_b
        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Error 
        error = abs(int_to_float(dut.data_out.value.integer, 1, 15) - int_to_float(expected, 1, 15))

        # Check the output.
        assert (
                error < 0.1
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 1, 15)}
            LUT: {int_to_float(data_b, 1, 15)}

            Output:
            Out: {int_to_float(dut.data_out.value.integer, 1, 15)}
            
            Expected: 
            {int_to_float(expected, 1, 15)}

            Error:
            {error}
            """

if __name__ == "__main__":
    mase_runner()
