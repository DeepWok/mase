#!/usr/bin/env python3

# This script tests the fixed point accumulator
import random, os, math, sys, logging

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

debug = True
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


class VerificationCase:
    def __init__(self, samples=10):
        self.samples = samples
        self.in_width = 32
        self.num = 9
        self.out_width = math.ceil(math.log2(self.num)) + self.in_width
        self.inputs = RandomSource(
            samples=self.samples * self.num,
            max_stalls=2 * self.samples,
            is_data_vector=False,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_DEPTH": self.num,
            "IN_WIDTH": self.in_width,
            "OUT_WIDTH": self.out_width,
        }

    def sw_compute(self):
        ref = []
        inputs = self.inputs.data
        for i in range(self.samples):
            output = 0
            for j in range(self.num):
                output += inputs[i * self.num + j]
            ref.append(output)
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
async def test_fixed_accumulator(dut):
    """Test accumulator"""
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
        # breakpoint()
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = ["../../../../components/fixed_arith/fixed_accumulator.sv"]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_accumulator",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="fixed_accumulator", test_module="fixed_accumulator_tb")


if __name__ == "__main__":
    runner()
