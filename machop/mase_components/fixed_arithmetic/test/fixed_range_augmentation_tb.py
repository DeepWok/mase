#!/usr/bin/env python3

# This script tests the fixed range augmentation used for the isqrt module.
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
    msb_index = width-1
    for i in range(1, width+1):
        power = 2 ** (width - i)
        if power <= x:
            msb_index = width - i
            break
    res = x
    if msb_index < (width - 1):
        res = res * 2 ** (width - 1 - msb_index)

    return res, msb_index

def range_augmentation_sw(x_red: int, msb_index: int, width: int, frac_width: int) -> int:
    const_len = 16
    ISQRT2 = float_to_int(1 / math.sqrt(2), 1, const_len-1)
    SQRT2 = float_to_int(math.sqrt(2), 1, const_len-1)
    """model of range augmentation for isqrt"""
    shifted_amount = frac_width - msb_index
    shift_amount = None
    res = None

    if shifted_amount > 0:
        if shifted_amount % 2 == 0:
            shift_amount = shifted_amount // 2
            res = x_red
        else:
            shift_amount = (shifted_amount - 1) // 2
            res = (x_red * SQRT2) >> (const_len - 1)
        res = res * 2 ** (shift_amount)
    elif shifted_amount < 0:
        if shifted_amount % 2 == 0:
            shift_amount = -shifted_amount // 2
            res = x_red
        else:
            shift_amount = (-shifted_amount - 1) // 2
            res = x_red * ISQRT2 // 2**(const_len - 1)
        res = res // 2 ** (shift_amount)
    else:
        res = x_red
    res = res >> (width - 1 - frac_width)
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
            "WIDTH",
            "FRAC_WIDTH"
        ])

    def generate_inputs(self):
        samples = 2 ** self.WIDTH
        data_x = []
        msb_indices = []
        for x in range(samples):
            x, msb_index = range_reduction_sw(x, self.WIDTH)
            data_x.append(x)
            msb_indices.append(msb_index)
        return data_x, msb_indices, samples
    def gen_single_test(self):
        return [3], [1], 1

    def model(self, data_x, msb_indices):
        ref = []
        for x, msb_index in zip(data_x, msb_indices):
            expected = range_augmentation_sw(x, msb_index, self.WIDTH, self.FRAC_WIDTH)
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_range_augmentation(dut):
    """Test for fixed range augmentation for isqrt"""
    testcase = VerificationCase(dut)
    data_x, msb_indices, samples = testcase.generate_inputs()
    #data_x, msb_indices, samples = testcase.gen_single_test()
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

if __name__ == "__main__":
    def full_sweep():
        parameter_list = []
        # TODO: model does not work for purely fractional numbers.
        for int_width in range(1, 9):
            for frac_width in range(0, 9):
                width = int_width + frac_width
                parameters = {
                        "WIDTH": width,
                        "FRAC_WIDTH": frac_width
                }
                parameter_list.append(parameters)
        return parameter_list

    parameter_list = full_sweep()
    #parameter_list = [{"WIDTH": 2, "FRAC_WIDTH": 2}]

    mase_runner(module_param_list=parameter_list)
