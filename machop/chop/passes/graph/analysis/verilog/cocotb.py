#!/usr/bin/env python3

import os, logging

from mase_cocotb.random_test import check_results
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

logger = logging.getLogger(__name__)


# DUT test specifications
class VerificationCase:
    def __init__(self, iterations=1, samples=10):
        self.samples = samples
        self.iterations = iterations


@cocotb.test()
async def test_top(dut):
    """Test top-level model hardware design"""
    samples = 1000
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
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 100):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in_0_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_0_ready.value = test_case.outputs.pre_compute(
            dut.data_out_0_valid.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.data_in_0_valid.value, dut.data_in_0.value = test_case.data_in.compute(
            dut.data_in_0_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_0_ready.value = test_case.outputs.compute(
            dut.data_out_0_valid.value, dut.data_out_0.value
        )
        debug_state(dut, "Pre-clk")

        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)
