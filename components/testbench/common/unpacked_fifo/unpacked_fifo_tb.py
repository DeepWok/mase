#!/usr/bin/env python3

# This script tests the fixed point dot product
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

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
        ref = self.data_in.data
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
        breakpoint()
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break

    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/ram_block.sv",
        "../../../../components/common/register_slice.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    for _ in range(1):
        runner.build(
            verilog_sources=verilog_sources,
            hdl_toplevel="unpacked_fifo",
            build_args=extra_args,
        )

    runner.test(hdl_toplevel="unpacked_fifo", test_module="unpacked_fifo_tb")


if __name__ == "__main__":
    runner()
