#!/usr/bin/env python3

# This script tests the binary multiplier
import random, os

import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner


def multiplier_sw(a: int, b: int) -> int:
    """model of multiplier"""
    return int(not (a ^ b))


@cocotb.test()
async def test_fixed_mult(dut):
    """Test for adding 2 random numbers multiple times"""

    for i in range(30):
        data_a = random.randint(0, 1)
        data_b = random.randint(0, 1)

        dut.data_a.value = data_a
        dut.data_b.value = data_b

        await Timer(10, units="ns")

        result = multiplier_sw(data_a, data_b)

        assert int(dut.product.value) == int(
            result
        ), "Randomised test failed with: {} * {} = {}, expect: {}".format(
            int(dut.data_a.value), int(dut.data_b.value), int(dut.product.value), result
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../binary_arith/binary_activation_binary_mult.sv"]

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources, hdl_toplevel="binary_activation_binary_mult"
    )
    runner.test(
        hdl_toplevel="binary_activation_binary_mult",
        test_module="binary_activation_binary_mult_tb",
    )


if __name__ == "__main__":
    runner()
