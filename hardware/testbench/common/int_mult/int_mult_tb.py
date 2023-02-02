# This script tests the fixed point multiplier
import random, os

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner


def multiplier_sw(a: int, b: int) -> int:
    """model of multiplier"""
    return a * b


@cocotb.test()
async def int_mult_test(dut):
    """Test for adding 2 random numbers multiple times"""

    for i in range(30):

        data_a = random.randint(0, 15)
        data_b = random.randint(0, 15)

        dut.data_a.value = data_a
        dut.data_b.value = data_b

        await Timer(10, units="ns")

        result = multiplier_sw(data_a, data_b)

        assert dut.product.value == result, "Randomised test failed with: {} * {} = {}, expect: {}".format(
            dut.data_a.value, dut.data_b.value, dut.product.value, result)


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../common/int_mult.sv"]

    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources, toplevel="int_mult")
    runner.test(toplevel="int_mult", py_module="int_mult_tb")


if __name__ == "__main__":
    runner()
