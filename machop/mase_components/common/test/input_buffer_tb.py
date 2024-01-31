#!/usr/bin/env python3

# This script tests the fixed point dot product
import logging

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
        self.in_width = 32
        self.in_parallelism = 4
        self.in_size = 2
        self.buffer_size = 32
        self.repeat = 4
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.buffer_size,
            num=self.in_parallelism * self.in_size,
            max_stalls=0,
            debug=debug,
        )
        self.outputs = RandomSink(
            name="output",
            samples=samples * self.repeat * self.buffer_size,
            max_stalls=0,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.in_width,
            "IN_PARALLELISM": self.in_parallelism,
            "IN_SIZE": self.in_size,
            "BUFFER_SIZE": self.buffer_size,
            "REPEAT": self.repeat,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            mid = []
            for j in range(self.buffer_size):
                mid.append(self.data_in.data[i * self.buffer_size + j])
            mid = mid * self.repeat
            ref += mid
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
async def test_fixed_dot_product(dut):
    """Test integer based vector mult"""
    samples = 100
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
    for i in range(samples * test_case.buffer_size * test_case.repeat * 10):
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
        # wave_check(dut)
        logger.debug("\n")
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break

    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of in_out:\n\
        {},{} data_in = {}\n\
        {},{} data_out = {}\n\
        ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )
    logger.debug(
        "wave of ram_buffer:\n\
        ram_buffer = {}\n\
        q0_t0      = {}".format(
            [int(i) for i in dut.ram_buffer.ram.value], int(dut.ram_buffer.q0_t0.value)
        )
    )


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()], trace=True)
