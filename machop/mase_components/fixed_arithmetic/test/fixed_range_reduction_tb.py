#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from mase_cocotb.testbench import Testbench
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math


## AIM: move the the MSB to the leftmost position.
def range_reduction_sw(x: int, width: int) -> int:
    """model of range reduction for isqrt"""
    # Find MSB
    # NOTE: if the input is 0 then consider msb index as width-1.
    msb_index = width-1
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

class VerificationCase(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)
        self.assign_self_params([
            "WIDTH"
        ])

    def generate_inputs(self):
        samples = 2 ** self.WIDTH
        return [val for val in range(0, samples)], samples
        
    def model(self, inputs):
        ref = []
        for x in inputs:
            expected = range_reduction_sw(x, self.WIDTH)
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_range_reduction(dut):
    """Test for adding 2 random numbers multiple times"""
    testcase = VerificationCase(dut)
    data_in_x, samples = testcase.generate_inputs()
    ref = testcase.model(data_in_x)

    for i in range(samples):
        # Set up module data.
        data_a = data_in_x[i]

        # Force module data.
        dut.data_a.value = data_a
        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = ref[i]

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
    def full_sweep():
        parameter_list = []
        for width in range(1, 17):
            parameters = {"WIDTH": width}
            parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()
    #parameter_list = [{"WIDTH": 2}]

    mase_runner(module_param_list=parameter_list)
