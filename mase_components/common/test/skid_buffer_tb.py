#!/usr/bin/env python3

# This script tests the register slice
import logging

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
        self.inputs = RandomSource(samples=samples, is_data_vector=False, debug=debug)
        self.outputs = RandomSink(samples=samples, debug=debug)
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
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_register_slice(dut):
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
    check_in = 0
    check_out = 0
    while not done:
        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")

        ## Pre_compute
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        ## Compute
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        in_out_wave(dut, "Pre-clk")
        logger.debug("\n")
        if dut.insert.value == 1:
            check_in += 1
        if dut.remove.value == 1:
            check_out += 1
        print("check_in = {}, check_out = {}".format(check_in, check_out))
        wave_check(dut)
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave_check:\n\
                {},{} data_in = {}\n\
                data_buffered_out = {}\n\
                {},{} data_out = {}\n\
                ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            int(dut.data_in.value),
            int(dut.data_buffer_out.value),
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            int(dut.data_out.value),
        )
    )


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
