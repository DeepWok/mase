#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner
import math
from mase_components.scalar_operators.fixed.test.isqrt_sw import (
    nr_stage_sw,
    float_to_int,
    int_to_float,
    fixed_lut_index_sw,
    make_lut,
    range_reduction_sw,
)


class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut)
        self.assign_self_params(["WIDTH"])

    def generate_inputs(self, lut_pow):
        samples = 2**self.WIDTH
        int_width = 1
        frac_width = self.WIDTH - 1
        data_x = []
        initial_guesses = []
        lut_size = 2**lut_pow
        lut = make_lut(lut_size, self.WIDTH)
        # NOTE: since negative values are not supported by fixed formats since
        # isqrt only outputs positive results we cannot test every single com-
        # bination of x and initial guesses.
        for x in range(samples):
            # Create inputs.
            x_red = range_reduction_sw(x, self.WIDTH)
            lut_index = fixed_lut_index_sw(x_red, self.WIDTH, lut_pow)
            lut_value = lut[lut_index]
            # Add inputs.
            data_x.append(x_red)
            initial_guesses.append(lut_value)
        return data_x, initial_guesses, samples

    def model(self, data_x, initial_guesses):
        ref = []
        for x, lut in zip(data_x, initial_guesses):
            expected = nr_stage_sw(x, self.WIDTH, lut)
            ref.append(expected)
        return ref


@cocotb.test()
async def cocotb_test_fixed_nr_stage(dut):
    """Test for the Newton Raphson stage for isqrt"""
    testcase = VerificationCase(dut)
    lut_pow = 5
    data_x, initial_guesses, samples = testcase.generate_inputs(lut_pow)
    ref = testcase.model(data_x, initial_guesses)
    width = testcase.WIDTH

    for i in range(samples):
        # Set up module data.
        data_a = data_x[i]
        data_b = initial_guesses[i]

        # Force module data.
        dut.data_a.value = data_a
        dut.data_b.value = data_b
        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = ref[i]

        # Error
        error = abs(
            int_to_float(dut.data_out.value.integer, 1, width)
            - int_to_float(expected, 1, width)
        )

        # Check the output.
        assert (
            error == 0
        ), f"""
            <<< --- Test failed --- >>>
            Input:
            X  : {int_to_float(data_a, 1, width-1)}
            LUT: {int_to_float(data_b, 1, width-1)}

            Output:
            Out: {int_to_float(dut.data_out.value.integer, 1, width-1)}

            Expected:
            {int_to_float(expected, 1, width-1)}

            Error:
            {error}
            """


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_nr_stage():
    def full_sweep():
        parameter_list = []
        for width in range(1, 17):
            parameter_list.append({"WIDTH": width})
        return parameter_list

    parameter_list = full_sweep()
    mase_runner(module_param_list=parameter_list)


if __name__ == "__main__":
    test_fixed_nr_stage()
