#!/usr/bin/env python3

# This script tests the fixed range augmentation used for the isqrt module.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner
import math
from mase_components.scalar_operators.fixed.test.isqrt_sw import (
    range_reduction_sw,
    range_augmentation_sw,
    int_to_float,
    find_msb,
)


class VerificationCase(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)
        self.assign_self_params(["WIDTH", "FRAC_WIDTH"])

    def generate_inputs(self):
        samples = 2**self.WIDTH
        data_x = []
        msb_indices = []
        for x in range(samples):
            x_red = range_reduction_sw(x, self.WIDTH)
            msb_index = find_msb(x, self.WIDTH)
            data_x.append(x_red)
            msb_indices.append(msb_index)
        return data_x, msb_indices, samples

    def model(self, data_x, msb_indices):
        ref = []
        for x, msb_index in zip(data_x, msb_indices):
            expected = range_augmentation_sw(x, msb_index, self.WIDTH, self.FRAC_WIDTH)
            ref.append(expected)
        return ref


@cocotb.test()
async def cocotb_test_fixed_range_augmentation(dut):
    """Test for fixed range augmentation for isqrt"""
    testcase = VerificationCase(dut)
    data_x, msb_indices, samples = testcase.generate_inputs()
    ref = testcase.model(data_x, msb_indices)
    int_width = testcase.WIDTH - testcase.FRAC_WIDTH
    frac_width = testcase.FRAC_WIDTH
    width = testcase.WIDTH

    for i in range(samples):
        # Set up module data.
        data_a = data_x[i]
        data_b = msb_indices[i]

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
            MSB: {data_b}

            Output:
            Out: {int_to_float(dut.data_out.value.integer, int_width, frac_width)}
            
            Expected: 
            {int_to_float(expected, int_width, frac_width)}
            """


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_range_augmentation():
    def full_sweep():
        parameter_list = []
        # TODO: model does not work for purely fractional numbers.
        for int_width in range(1, 9, 3):
            for frac_width in range(0, 9, 3):
                width = int_width + frac_width
                parameters = {"WIDTH": width, "FRAC_WIDTH": frac_width}
                parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()
    # parameter_list = [{"WIDTH": 2, "FRAC_WIDTH": 2}]

    mase_runner(module_param_list=parameter_list)


if __name__ == "__main__":
    test_fixed_range_augmentation()
