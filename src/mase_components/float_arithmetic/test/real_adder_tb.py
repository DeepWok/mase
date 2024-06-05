#!/usr/bin/env python3

# This script tests the fixed point multiplier
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner

import struct


def add_sw(a: float, b: float) -> int:
    """model of multiplier"""
    return a + b


def float2int(x: float):
    return struct.unpack("!I", struct.pack("!f", x))[0]


def int2float(x: int):
    return struct.unpack("f", struct.pack("i", x))[0]


@cocotb.test()
async def cocotb_test_real_adder(dut):
    """Test for adding 2 random numbers multiple times"""

    for i in range(30):
        in1 = random.random()
        in2 = random.random()
        # in1 = 2.0
        # in2 = 2.0

        hw_in1 = float2int(in1)
        hw_in2 = float2int(in2)
        dut.in1.value = hw_in1
        dut.in2.value = hw_in2

        await Timer(10, units="ns")
        hw_result = int2float(int(dut.res.value))
        sw_result = add_sw(int2float(int(dut.in1.value)), int2float(int(dut.in2.value)))
        check = float("{:.6}".format(hw_result)) == float("{:.6}".format(sw_result))
        # breakpoint()
        assert check, "Randomised test failed with: {} + {} = {}, expect: {}".format(
            int2float(int(dut.in1.value)),
            int2float(int(dut.in2.value)),
            int2float(int(dut.res.value)),
            sw_result,
        )


def test_real_adder():
    mase_runner()


if __name__ == "__main__":
    test_real_adder()
