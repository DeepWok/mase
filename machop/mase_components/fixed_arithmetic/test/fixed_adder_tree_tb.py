#!/usr/bin/env python3

# This script tests the fixed point adder tree
import os, math, logging

from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.data_in_width = 32
        self.num = 9
        self.data_out_width = math.ceil(math.log2(self.num)) + 32
        self.inputs = RandomSource(
            samples=samples, num=self.num, max_stalls=2 * samples, debug=debug
        )
        self.outputs = RandomSink(
            samples=samples, num=self.num, max_stalls=2 * samples, debug=debug
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_SIZE": self.num,
            "IN_WIDTH": self.data_in_width,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(sum(self.inputs.data[i]))
        ref.reverse()
        return ref


# Check if an impossible state is reached
def is_impossible_state(data_in_ready, data_in_valid, data_out_ready, data_out_valid):
    # (0, X, 0, 0)
    # (0, X, 1, 0)
    # (0, X, 1, 1)
    if (not data_in_ready) and not ((not data_out_ready) and data_out_valid):
        return True
    return False


@cocotb.test()
async def test_fixed_adder_tree(dut):
    """Test integer based adder tree"""
    samples = 20
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Synchronize with the clock
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    logger.debug(
        "Pre-clk  State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    await FallingEdge(dut.clk)
    logger.debug(
        "Post-clk State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    logger.debug(
        "Pre-clk  State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    await FallingEdge(dut.clk)
    logger.debug(
        "Post-clk State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )

    done = False
    while not done:
        await FallingEdge(dut.clk)
        logger.debug(
            "Post-clk State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        logger.debug(
            "Pre-clk  State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
