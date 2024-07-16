#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner
import math
from mase_components.scalar_operators.fixed.test.isqrt_sw import (
    find_msb,
    range_reduction_sw,
    fixed_lut_index_sw,
    int_to_float,
)


class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut)
        self.assign_self_params(["WIDTH", "LUT_POW"])

    def generate_inputs(self):
        samples = 2**self.WIDTH
        data_x = []
        msb_indices = []
        for x in range(samples):
            x_red = range_reduction_sw(x, self.WIDTH)
            data_x.append(x_red)
            msb_indices.append(find_msb(x, self.WIDTH))

        return data_x, msb_indices, samples

    def gen_single_test(self):
        return [127], [find_msb(127, self.WIDTH)], 1

    def model(self, data_x, msb_indices):
        ref = []
        for x_red, msb_index in zip(data_x, msb_indices):
            expected = fixed_lut_index_sw(x_red, self.WIDTH, self.LUT_POW)
            ref.append(expected)
        return ref


def debug(dut):
    print("X:   ", dut.data_a.value)
    print("MSB: ", dut.data_b.value.integer)
    print("Temp : ", dut.temp.value)
    print("Out: ", dut.data_out.value.integer)


@cocotb.test()
async def cocotb_test_fixed_lut_index(dut):
    """Test for finding LUT index for ISQRT"""
    testcase = VerificationCase(dut)
    width = testcase.WIDTH
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


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_lut_index():

    def full_sweep():
        parameter_list = []
        lut_pow = 5
        for width in range(1, 17):
            parameters = {"WIDTH": width, "LUT_POW": lut_pow}
            parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()
    # parameter_list = [{"WIDTH": 7, "LUT_POW": 5}]

    mase_runner(module_param_list=parameter_list)


if __name__ == "__main__":
    test_fixed_lut_index()
