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

def isqrt_sw(x: int, int_width: int, frac_width: int) -> int:
    """model of fixed point isqrt"""
    if x == 0:
        return 2 ** (int_width + frac_width) - 1
    x_f = int_to_float(x, int_width, frac_width)
    ref = 1 / math.sqrt(x_f)
    return ref

class VerificationCase:
    def __init__(self, samples=1, in_width=16, frac_width=8):
        self.in_width       = in_width
        self.in_frac_width  = frac_width
        self.lut_pow        = 5
        self.out_width      = in_width
        self.out_frac_width = frac_width
        self.out_frac_width = self.in_frac_width
        self.pipeline_cycles = 0
        self.data_in = [val for val in range(samples)]
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.in_width,
            "IN_FRAC_WIDTH": self.in_frac_width,
            "LUT_POW": self.lut_pow,
            "OUT_WIDTH": self.out_width,
            "OUT_FRAC_WIDTH": self.out_frac_width,
            "PIPELINE_CYCLES": self.pipeline_cycles
        }

    def sw_compute(self):
        ref = []
        for sample in self.data_in:
            int_width = self.in_width - self.in_frac_width
            expected = isqrt_sw(sample, int_width, self.in_frac_width)
            ref.append(expected)
        return ref


# TODO: why do the parameters not get updated???
@cocotb.test()
async def test_fixed_isqrt(dut):
    """Test for inverse square root"""
    in_width = 16
    frac_width = 8
    int_width = in_width - frac_width
    samples = 2**(int_width + frac_width) - 1
    testcase = VerificationCase(samples, in_width, frac_width)
    print(testcase.get_dut_parameters())

    for i in range(samples):
        # Set up module data.
        data_a = testcase.data_in[i]

        # Force module data.
        dut.in_data.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Check the output.
        assert (
                int_to_float(dut.out_data.value.integer, int_width, frac_width) - expected < 2**(-8)
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X float : {int_to_float(data_a, int_width, frac_width)}

            Output:
            {int_to_float(dut.out_data.value.integer, int_width, frac_width)}
            
            Expected: 
            {expected}

            Test index:
            {i}
            """


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
