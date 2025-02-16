#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os
from pathlib import Path
from os import makedirs

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import verilator_str_param, bit_driver
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_components.scalar_operators.fixed.test.isqrt_sw import (
    isqrt_sw2,
    int_to_float,
    make_lut,
)
from mase_components.common.test.lut_tb import write_memb


class VerificationCase(Testbench):
    def __init__(self, dut):
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "LUT_POW",
            ]
        )

        self.input_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            name="Output ISQRT",
        )

    def generate_inputs(self, num=10000):
        maxnum = (2**self.IN_WIDTH) - 1
        return [random.randint(0, maxnum) for _ in range(num)], num

    def model(self, data_in):
        ref = []
        lut_size = 2**self.LUT_POW
        lut = make_lut(lut_size, self.IN_WIDTH)
        for x in data_in:
            expected = isqrt_sw2(
                x, self.IN_WIDTH, self.IN_FRAC_WIDTH, self.LUT_POW, lut
            )
            ref.append(expected)
        return ref


def debug(dut, i, f):
    print(
        f"X        : {dut.in_data.value}  {int_to_float(dut.in_data.value.integer, i, f)}"
    )
    print(
        f"X red    : {dut.x_reduced.value}    {int_to_float(dut.x_reduced.value.integer, 1, 15)}"
    )
    print(f"MSB index: {dut.msb_index.value.integer}")
    print(f"lut_index: {dut.lut_index.value.integer}")
    print(
        f"LUT value: {dut.lut_value.value}    {int_to_float(dut.lut_value.value.integer, 1, 15)}"
    )
    print(f"Y        : {dut.y.value}    {int_to_float(dut.y.value.integer, 1, 15)}")
    print(
        f"Y aug    : {dut.y_aug.value}    {int_to_float(dut.y_aug.value.integer, i, f)}"
    )


CLK_NS = 25


@cocotb.test()
async def sweep(dut):
    """Test for inverse square root"""
    tb = VerificationCase(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    inputs, samples = tb.generate_inputs()
    exp_out = tb.model(inputs)
    tb.input_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def backpressure(dut):
    """Test for inverse square root"""
    tb = VerificationCase(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    inputs, samples = tb.generate_inputs()
    exp_out = tb.model(inputs)
    tb.input_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_backpressure(dut):
    """Test for inverse square root"""
    tb = VerificationCase(dut)
    await tb.reset()
    tb.input_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    inputs, samples = tb.generate_inputs()
    exp_out = tb.model(inputs)
    tb.input_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_isqrt():
    mem_dir = Path(__file__).parent / "build" / "fixed_isqrt" / "mem"
    makedirs(mem_dir, exist_ok=True)

    def single_cfg(width, frac_width, lut_pow, str_id):
        lut_size = 2**lut_pow
        lut = make_lut(lut_size, width)
        mem_path = mem_dir / f"lutmem-{str_id}.mem"
        write_memb(mem_path, lut, width)
        return {
            "IN_WIDTH": width,
            "IN_FRAC_WIDTH": frac_width,
            "LUT_POW": lut_pow,
            "LUT_MEMFILE": verilator_str_param(str(mem_path)),
        }

    def full_sweep():
        parameter_list = []
        lut_pow = 5
        for int_width in range(1, 9):
            for frac_width in range(0, 9):
                width = int_width + frac_width
                parameters = single_cfg(
                    width, frac_width, lut_pow, str_id=f"{int_width}-{frac_width}"
                )
                parameter_list.append(parameters)
        return parameter_list

    parameter_list = [
        # A use case in group_norm
        *full_sweep(),
        # single_cfg(35, 14, 7, 0)
    ]
    mase_runner(module_param_list=parameter_list, trace=True)


if __name__ == "__main__":
    test_fixed_isqrt()
