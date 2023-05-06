#!/usr/bin/env python3

# This script tests the fixed point multiplier
import random, os

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner


def multiplier_sw(a: int, b: int) -> int:
    """model of multiplier"""
    return a * b


@cocotb.test()
async def test_fixed_mult(dut):
    """Test for adding 2 random numbers multiple times"""

    for i in range(30):
        data_a = random.randint(0, 15)
        data_b = random.randint(0, 15)

        dut.data_a.value = data_a
        dut.data_b.value = data_b

        await Timer(10, units="ns")

        result = multiplier_sw(data_a, data_b)

        assert (
            dut.product.value == result
        ), "Randomised test failed with: {} * {} = {}, expect: {}".format(
            int(dut.data_a.value), int(dut.data_b.value), int(dut.product.value), result
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../fixed_arith/fixed_mult.sv"]

    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources, toplevel="fixed_mult")
    runner.test(toplevel="fixed_mult", py_module="fixed_mult_tb")


if __name__ == "__main__":
    runner()
