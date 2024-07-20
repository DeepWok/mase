#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from mase_cocotb.testbench import Testbench
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math
from mase_components.scalar_operators.fixed.test.isqrt_sw import (
    range_reduction_sw,
    int_to_float,
)


class VerificationCase(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)
        self.assign_self_params(["WIDTH"])

    def generate_inputs(self):
        samples = 2**self.WIDTH
        return [val for val in range(0, samples)], samples

    def model(self, inputs):
        ref = []
        for x in inputs:
            expected = range_reduction_sw(x, self.WIDTH)
            ref.append(expected)
        return ref


@cocotb.test()
async def cocotb_test_fixed_range_reduction(dut):
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


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_range_reduction():

    def full_sweep():
        parameter_list = []
        for width in range(1, 17):
            parameters = {"WIDTH": width}
            parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()

    mase_runner(module_param_list=parameter_list)


if __name__ == "__main__":
    test_fixed_range_reduction()
