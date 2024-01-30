#!/usr/bin/env python3

# This script tests the fixed point multiplier
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner


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


if __name__ == "__main__":
    mase_runner()
