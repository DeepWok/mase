#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner
import math

def find_msb_index(x: int, width: int) -> int:
    msb_index = width-1
    for i in range(1, width+1):
        power = 2 ** (width - i)
        if power <= x:
            msb_index = width - i
            break
    return msb_index

def fixed_lut_index_sw(x: int, msb_index: int, width: int, lut_pow: int) -> int:
    """model for finding the lut index for lut isqrt value"""
    res = x * 2 ** (width - 1 - msb_index)
    res = res - 2 ** (width - 1)
    res = res * 2 ** lut_pow
    res = res / 2 ** (width - 1)
    # FORMAT OUTPUT: Q(WIDTH).0
    return int(res)

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

class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut)
        self.assign_self_params([
            "WIDTH",
            "LUT_POW"
        ])
        self.data_in_x = [i for i in range(1, samples+1)]
        self.data_in_msb_index = []
        for x in self.data_in_x:
            self.data_in_msb_index.append(find_msb_index(x, self.data_in_width))

        self.samples = samples
        self.lut_pow = 5
        self.ref = self.sw_compute()

    def generate_inputs(self):
        samples = 2 ** self.WIDTH
        data_x = [i for i in range(samples)]
        msb_indices = []
        for x in data_x:
            msb_indices.append(find_msb_index(x, self.WIDTH))
        return data_x, msb_indices, samples

    def model(self, data_x, msb_indices):
        ref = []
        for x, msb_index in zip(data_x, msb_indices):
            expected = fixed_lut_index_sw(x, msb_index, self.WIDTH, self.LUT_POW)
            ref.append(expected)
        return ref

def debug(dut):
    print("X:   ", dut.data_a.value)
    print("MSB: ", dut.data_b.value.integer)
    print("Out: ", dut.data_out.value.integer)

@cocotb.test()
async def test_fixed_lut_index(dut):
    """Test for finding LUT index for ISQRT"""
    testcase = VerificationCase(dut)
    data_in_x, data_in_msb_index, samples = testcase.generate_inputs()
    ref = testcase.model(data_in_x, data_in_msb_index)

    for i in range(samples):
        # Set up module data.
        data_a = data_in_x[i]
        data_b = data_in_msb_index[i]

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
            X  : {int_to_float(data_a, 16, 0)}

            Output:
            Out: {dut.data_out.value.integer}
            
            Expected: 
            {expected}
            """

if __name__ == "__main__":
    mase_runner()
