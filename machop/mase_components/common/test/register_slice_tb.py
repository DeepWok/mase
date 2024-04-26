#!/usr/bin/env python3

# This script tests the register slice
import os, logging

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
        self.data_width = 32
        self.inputs = RandomSource(
            samples=samples, max_stalls=2 * samples, is_data_vector=False, debug=debug
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_width,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(self.inputs.data[i])
        ref.reverse()
        return ref


def in_out_wave(dut, name):
    logger.debug(
        "{}  State: (in_valid,in_ready,out_valid,out_ready) = ({},{},{},{})".format(
            name,
            dut.in_ready.value,
            dut.in_valid.value,
            dut.out_ready.value,
            dut.out_data.value,
        )
    )


@cocotb.test()
async def cocotb_test_register_slice(dut):
    """Test register slice"""
    samples = 30
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
    dut.in_valid.value = 0
    dut.out_ready.value = 1
    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    done = False
    while not done:
        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")

        ## Pre_compute
        dut.in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.out_ready.value = test_case.outputs.pre_compute(dut.out_data.value)
        await Timer(1, units="ns")

        ## Compute
        dut.in_valid.value, dut.in_data.value = test_case.inputs.compute(
            dut.in_ready.value
        )
        await Timer(1, units="ns")
        dut.out_ready.value = test_case.outputs.compute(
            dut.out_data.value, dut.out_data.value
        )
        in_out_wave(dut, "Pre-clk")
        logger.debug("\n")
        # breakpoint()
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_register_slice():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_register_slice()
