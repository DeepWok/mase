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
        self.wrap_y = 3
        self.in_y = 5
        self.in_x = 8
        self.unroll_in_x = 4
        self.iter_x = self.in_x // self.unroll_in_x
        # random.seed(0)
        self.inputs = RandomSource(
            samples=samples * self.in_y * self.iter_x,
            num=self.unroll_in_x,
            max_stalls=2 * samples,
            is_data_vector=True,
            debug=debug,
        )
        self.wraps = RandomSource(
            samples=samples * self.wrap_y * self.iter_x,
            num=self.unroll_in_x,
            max_stalls=2 * samples,
            is_data_vector=True,
            debug=debug,
        )
        self.outputs = RandomSink(
            samples=samples * (self.in_y + self.wrap_y) * self.iter_x,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_width,
            "IN_Y": self.in_y,
            "WRAP_Y": self.wrap_y,
            "IN_X": self.in_x,
            "UNROLL_IN_X": self.unroll_in_x,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            k = self.samples - 1 - i
            wrap = []
            data_in = []
            for j in range(self.wrap_y * self.iter_x):
                m = self.wrap_y * self.iter_x - 1 - j
                wrap.append(self.wraps.data[k * self.wrap_y * self.iter_x + m])
            for j in range(self.in_y * self.iter_x):
                m = self.in_y * self.iter_x - 1 - j
                data_in.append(self.inputs.data[k * self.in_y * self.iter_x + m])
            ref = ref + wrap + data_in
        print(ref)
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
    dut.wrap_in_valid.value = 0
    dut.data_out_ready.value = 1
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
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.wrap_in_valid.value = test_case.wraps.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        ## Compute
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        dut.wrap_in_valid.value, dut.wrap_in.value = test_case.wraps.compute(
            dut.wrap_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        in_out_wave(dut, "Pre-clk")
        wave_check(dut)
        # breakpoint()
        done = (
            test_case.inputs.is_empty()
            and test_case.wraps.is_empty()
            and test_case.outputs.is_full()
        )

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave_check:\n\
                {},{} data_in = {}\n\
                {},{} wrap_in = {}\n\
                {},{} data_out = {}\n\
                ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.wrap_in_valid.value,
            dut.wrap_in_ready.value,
            [int(i) for i in dut.wrap_in.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
