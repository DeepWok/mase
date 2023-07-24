#!/usr/bin/env python3

# This script tests the fixed point multiplier
import random, os

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner


@cocotb.test()
async def test_fixed_mult(dut):
    """Test for adding 2 random numbers multiple times"""

    for i in range(30):
        data_a = random.randint(-15, 15)

        dut.data_a.value = data_a

        await Timer(10, units="ns")

        assert (
            dut.product.value == data_a
        ), "Randomised test failed with: {} = {}, expect: {}".format(
            int(dut.data_a.value), int(dut.product.value), data_a
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../common/test_tb.sv"]

    runner = get_runner(sim)
    runner.build(verilog_sources=verilog_sources, hdl_toplevel="test_tb")
    runner.test(hdl_toplevel="test_tb", test_module="test_tb_tb")


if __name__ == "__main__":
    runner()
