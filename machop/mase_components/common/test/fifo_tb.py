#!/usr/bin/env python3

# This script tests the fixed point dot product
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
        self.depth = 8
        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=1,
            max_stalls=0,
            is_data_vector=False,
            debug=debug,
        )
        self.outputs = RandomSink(
            name="output",
            samples=samples,
            max_stalls=0,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DEPTH": self.depth,
            "DATA_WIDTH": self.data_width,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(self.data_in.data[i])
        ref.reverse()
        return ref


# Check if an is_impossible state is reached
def debug_state(dut, state):
    logger.debug(
        "{} State: (in_ready,in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fifo(dut):
    """Test integer based vector mult"""
    samples = 20
    test_case = VerificationCase(samples=samples)

    # Reset cycles
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

    await Timer(1, units="ns")

    await FallingEdge(dut.clk)

    await FallingEdge(dut.clk)

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 20):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        logger.debug("\n")
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break

    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"
    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
