#!/usr/bin/env python3

# This script tests the binary multiplier
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import utils

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner
from cocotb.binary import BinaryRepresentation, BinaryValue

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


def multiplier_sw(a: int, b: int, in_width: int) -> int:
    """model of multiplier"""
    assert b in [0, 1]
    return (a if b else -a) % (1 << in_width)


IN_WIDTH = 16


@cocotb.test()
async def test_fixed_mult(dut):
    """Test for adding 2 random numbers multiple times"""
    for i in range(30):
        data_a = random.randint(-15, 15)
        data_b = random.randint(0, 1)

        dut.data_a.value = data_a
        dut.data_b.value = data_b

        await Timer(10, units="ns")

        result = multiplier_sw(data_a, data_b, IN_WIDTH)

        assert int(dut.product.value) == int(
            result
        ), "Randomised test failed with: {} * {} = {}, expect: {}".format(
            int(dut.data_a.value),
            utils.binary_decode(int(dut.data_b.value)),
            dut.product.value.signed_integer,
            int(result),
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../binary_arith/fixed_activation_binary_mult.sv"]

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_activation_binary_mult",
        build_args=[f"-GIN_A_WIDTH={IN_WIDTH}"],
    )
    runner.test(
        hdl_toplevel="fixed_activation_binary_mult",
        test_module="fixed_activation_binary_mult_tb",
    )


if __name__ == "__main__":
    runner()
