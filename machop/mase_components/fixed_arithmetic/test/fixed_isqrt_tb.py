#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math
from mase_cocotb.testbench import Testbench
from isqrt_sw import isqrt_sw2, int_to_float, make_lut


class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut)
        self.assign_self_params([
            "IN_WIDTH",
            "IN_FRAC_WIDTH",
            "LUT_POW",
            "OUT_WIDTH",
            "OUT_FRAC_WIDTH"
        ])

    def generate_inputs(self):
        samples = 2 ** self.IN_WIDTH
        return [val for val in range(samples)], samples

    def model(self, data_in):
        ref = []
        lut_size = 2 ** self.LUT_POW
        lut = make_lut(lut_size, self.IN_WIDTH)
        for x in data_in:
            expected = isqrt_sw2(x, self.IN_WIDTH, self.IN_FRAC_WIDTH, self.LUT_POW, lut)
            ref.append(expected)
        return ref

def debug(dut, i, f):
    print(f"X        : {dut.in_data.value}  {int_to_float(dut.in_data.value.integer, i, f)}")
    print(f"X red    : {dut.x_reduced.value}    {int_to_float(dut.x_reduced.value.integer, 1, 15)}")
    print(f"MSB index: {dut.msb_index.value.integer}")
    print(f"lut_index: {dut.lut_index.value.integer}")
    print(f"LUT value: {dut.lut_value.value}    {int_to_float(dut.lut_value.value.integer, 1, 15)}")
    print(f"Y        : {dut.y.value}    {int_to_float(dut.y.value.integer, 1, 15)}")
    print(f"Y aug    : {dut.y_aug.value}    {int_to_float(dut.y_aug.value.integer, i, f)}")


@cocotb.test()
async def test_fixed_isqrt(dut):
    """Test for inverse square root"""
    testcase = VerificationCase(dut)
    data_in, samples = testcase.generate_inputs()
    ref = testcase.model(data_in)

    for i in range(samples):
        # Set up module data.
        data_a = data_in[i]

        # Force module data.
        dut.in_data.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        #debug(dut, int_width, frac_width)

        # Exepected result.
        expected = ref[i]

        # Check the output.
        assert (
                expected == dut.out_data.value.integer
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
    def create_lut_parameters(lut_size, width):
        lut_parameters = {}
        lut = make_lut(lut_size, width)
        lut_prefix = "LUT"
        for i in range(lut_size):
            if i < 10:
                lut_suffix = "0" + str(i)
            else:
                lut_suffix = str(i)
            name = lut_prefix + lut_suffix
            lut_parameters |= {name: lut[i]}
        return lut_parameters

    def single_test(width, frac_width, lut_pow, pipeline_cycles):
        parameter_list = []
        lut_size = 2 ** lut_pow
        parameters = {
                "IN_WIDTH": width, "IN_FRAC_WIDTH": frac_width,
                "OUT_WIDTH": width, "OUT_FRAC_WIDTH": frac_width,
                "LUT_POW": lut_pow, "PIPELINE_CYCLES": pipeline_cycles
                }
        lut_parameters = create_lut_parameters(lut_size, width)
        parameter_list.append(parameters | lut_parameters)
        return parameter_list

    def full_sweep():
        parameter_list = []
        lut_pow = 5
        lut_size = 2 ** lut_pow
        pipeline_cycles = 0
        for int_width in range(1, 9):
            for frac_width in range(0, 9):
                width = int_width + frac_width
                parameters = single_test(width, frac_width, lut_pow, pipeline_cycles)
                parameter_list += parameters
        return parameter_list

    parameter_list = full_sweep()

    #parameter_list = single_test(8, 4)

    mase_runner(module_param_list=parameter_list, trace=False)
