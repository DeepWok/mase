#!/usr/bin/env python3

# This script tests the register slice
import os, logging

from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = True
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.data_width = 32
        self.inputs = RandomSource(
            samples=samples, max_stalls=samples / 2, is_data_vector=False, debug=debug
        )
        self.outputs = RandomSink(samples=samples, max_stalls=samples / 2, debug=debug)
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
        "{}  State: (data_in,data_in_valid,data_in_ready) = ({},{},{})".format(
            name,
            int(dut.data_in.value),
            int(dut.data_in_valid.value),
            int(dut.data_in_ready.value),
        )
    )
    logger.debug(
        "{}  State: (data_out,data_out_valid,data_out_ready) = ({},{},{})".format(
            name,
            int(dut.data_out.value),
            int(dut.data_out_valid.value),
            int(dut.data_out_ready.value),
        )
    )
    logger.debug(
        "{}  State: (shift_reg, buffer) = ({},{})".format(
            name,
            int(dut.shift_reg.value),
            int(dut.buffer.value),
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
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    done = False
    test_limit = 100
    i = 0
    while not done and i < test_limit:

        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")

        ## Pre_compute
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        in_out_wave(dut, "pre-comput")

        ## Compute
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")

        in_out_wave(dut, "Pre-clk")
        logger.debug("\n")

        done = test_case.inputs.is_empty() and test_case.outputs.is_full()
        i += 1

    print([int(k) for k in test_case.outputs.data])
    print(test_case.ref)
    check_results(test_case.outputs.data, test_case.ref)


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_register_slice():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_register_slice()
