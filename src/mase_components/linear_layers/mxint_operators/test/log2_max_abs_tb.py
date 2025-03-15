#!/usr/bin/env python3

# This script tests the fixed point adder tree
import os, math, logging, pytest

from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.testbench import Testbench
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
class VerificationCase(Testbench):
    def __init__(self, dut, samples=10):
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["IN_SIZE", "IN_WIDTH"])
        self.data_in_width = self.IN_WIDTH
        self.num = self.IN_SIZE
        self.inputs = RandomSource(
            samples=samples, num=self.num, max_stalls=2 * samples, debug=debug
        )
        self.outputs = RandomSink(
            samples=samples, num=self.num, max_stalls=2 * samples, debug=debug
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(
                math.ceil(math.log2(max([abs(data) for data in self.inputs.data[i]])))
            )
            print(self.inputs.data[i])
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
async def cocotb_test_fixed_adder_tree(dut):
    """Test integer based adder tree"""
    samples = 1
    test_case = VerificationCase(dut, samples=samples)

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
    check_results([i.signed_integer for i in test_case.outputs.data], test_case.ref)


@pytest.mark.dev
def test_abs_max_tree():
    mase_runner(
        module_param_list=[
            # Power of 2's
            {"IN_SIZE": 2, "IN_WIDTH": 8},
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_abs_max_tree()
