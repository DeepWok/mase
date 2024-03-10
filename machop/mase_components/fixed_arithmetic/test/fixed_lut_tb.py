#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
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

    return lut

def lut_value_sw(lut, lut_index: int, lut_size: int) -> int:
    return lut[lut_index%lut_size]

class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut)
        self.assign_self_params([
            "WIDTH", "LUT_POW", "LUT00", "LUT01", "LUT02", "LUT03",
            "LUT04", "LUT05", "LUT06", "LUT07", "LUT08", "LUT09",
            "LUT10", "LUT11", "LUT12", "LUT13", "LUT14", "LUT15",
            "LUT16", "LUT17", "LUT18", "LUT19", "LUT20", "LUT21",
            "LUT22", "LUT23", "LUT24", "LUT25", "LUT26", "LUT27",
            "LUT28", "LUT29", "LUT30", "LUT31"
        ])

    def generate_inputs(self):
        samples = 2 ** self.LUT_POW
        data_x = [x for x in range(samples)]
        return data_x, samples

    def model(self, data_x):
        lut_size = 2 ** self.LUT_POW
        lut = make_lut(lut_size, self.WIDTH)
        ref = []
        for x in data_x:
            expected = lut_value_sw(lut, x, lut_size)
            ref.append(expected)
        return ref

@cocotb.test()
async def test_fixed_lut(dut):
    """Test for finding LUT value for ISQRT"""
    testcase = VerificationCase(dut)
    data_x, samples = testcase.generate_inputs()
    ref = testcase.model(data_x)

    for i in range(samples):
        # Set up module data.
        data_a = data_x[i]

        # Force module data.
        dut.data_a.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = ref[i]

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
    def full_sweep():    
        parameter_list = []
        lut_pow = 5      
        lut_size = 2 ** lut_pow
        for width in range(1, 17):
            lut = make_lut(lut_size, width)
            parameters = {"WIDTH": width, "LUT_POW": lut_pow}
            lut_prefix = "LUT"
            for i in range(lut_size):
                if i < 10:
                    lut_suffix = "0" + str(i)
                else:
                    lut_suffix = str(i)
                name = lut_prefix + lut_suffix
                parameters |= {name: lut[i]}
            parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()

    mase_runner(module_param_list=parameter_list)
