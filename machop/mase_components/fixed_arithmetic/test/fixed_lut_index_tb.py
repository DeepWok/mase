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

def range_reduction_sw(x: int, width: int) -> int:
    """model of range reduction for isqrt"""
    # Find MSB
    # NOTE: if the input is 0 then consider msb index as width-1.
    msb_index = find_msb_index(x, width)
    res = x
    if msb_index < (width - 1):
        res = res * 2 ** (width - 1 - msb_index)
    return res

def fixed_lut_index_sw(x_red: int, msb_index: int, width: int, lut_pow: int) -> int:
    """model for finding the lut index for lut isqrt value"""
    if width == 1 or x_red == 0:
        res = 0
    else:
        res = x_red - 2 ** (width - 1)
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

    def generate_inputs(self):
        samples = 2 ** self.WIDTH
        data_x = []
        msb_indices = []
        for x in range(samples):
            x_red = range_reduction_sw(x, self.WIDTH)
            data_x.append(x_red)
            msb_indices.append(find_msb_index(x, self.WIDTH))

        return data_x, msb_indices, samples

    def gen_single_test(self):
        return [127], [find_msb_index(127, self.WIDTH)], 1

    def model(self, data_x, msb_indices):
        ref = []
        for x_red, msb_index in zip(data_x, msb_indices):
            expected = fixed_lut_index_sw(x_red, msb_index, self.WIDTH, self.LUT_POW)
            ref.append(expected)
        return ref

def debug(dut):
    print("X:   ", dut.data_a.value)
    print("MSB: ", dut.data_b.value.integer)
    print("Temp : ", dut.temp.value)
    print("Out: ", dut.data_out.value.integer)

@cocotb.test()
async def test_fixed_lut_index(dut):
    """Test for finding LUT index for ISQRT"""
    testcase = VerificationCase(dut)
    width = testcase.WIDTH
    data_in_x, data_in_msb_index, samples = testcase.generate_inputs()
    #data_in_x, data_in_msb_index, samples = testcase.gen_single_test()
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

        #debug(dut)

        # Exepected result.
        expected = ref[i]

        # Check the output.
        assert (
                dut.data_out.value.integer == expected
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 1, width-1)}

            Output:
            Out: {dut.data_out.value.integer}
            
            Expected: 
            {expected}
            """

if __name__ == "__main__":
    def full_sweep():
        parameter_list = []
        lut_pow = 5
        for width in range(1, 17):
            parameters = {
                    "WIDTH": width,
                    "LUT_POW": lut_pow
            }
            parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()
    #parameter_list = [{"WIDTH": 7, "LUT_POW": 5}]

    mase_runner(module_param_list=parameter_list)
