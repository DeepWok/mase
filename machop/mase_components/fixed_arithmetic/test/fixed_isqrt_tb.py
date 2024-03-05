#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math

# TODO: for 8 bits only. Need to generalise.
MAX_NUM = 255

def isqrt_sw(x: float) -> int:
    """model of multiplier"""
    if x == 0:
        print("MAX NUM:", MAX_NUM)
        return MAX_NUM
    res = 1 / math.sqrt(x)
    return int(res)

# TODO: this is just for the Q8.0 format. Need to generalise to other formats.
class VerificationCase:
    def __init__(self, samples=1):
        self.data_in_width = 8
        self.data_in = [0, 1]
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width
        }

    def sw_compute(self):
        ref = []
        for sample in self.data_in:
            expected = isqrt_sw(sample) 
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_isqrt(dut):
    """Test for adding 2 random numbers multiple times"""
    testcase = VerificationCase()

    #for i in range(samples):
    for i in range(len(testcase.data_in)):
        # Set up module data.
        data_a = testcase.data_in[i]

        # Force module data.
        dut.data_a.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Check the output.
        assert (
            dut.isqrt.value == expected
        ), "Randomised test failed with: 1/sqrt({}) = {}, expect: {}".format(
            int(dut.data_a.value), int(dut.isqrt.value), expected
        )


if __name__ == "__main__":
    mase_runner()
