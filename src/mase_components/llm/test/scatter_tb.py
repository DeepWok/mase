#!/usr/bin/env python3


# This script tests the scatter module

# Manually add user-specific mase_cocotb path
# this should be ignored on the standard mase-docker env
import sys, pytest

p = "/home/ic/MYWORKSPACE/Mase-DeepWok/machop"
sys.path.append(p)
##################################################

import os, math, logging

from mase_cocotb.random_test import *
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
        self.data_in_width = 16
        self.data_in_frac_width = 0
        self.in_rows = 5
        self.in_columns = 4
        self.max_large_numbers = 3
        self.large_number_thres = 127  # number larger than (BUT NOT EUQAL TO) threshold are counted as outliers

        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.in_rows * self.in_columns,
            max_stalls=0,
            debug=debug,
            arithmetic="llm-fp16-datain",
        )

        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)

        self.samples = samples
        self.ref = self.sw_compute()
        # self.ref = self.sw_cast(
        #     inputs=self.ref,
        #     in_width=self.data_in_width,
        #     in_frac_width=self.data_in_frac_width,
        #     out_width=self.data_in_width,
        #     out_frac_width=self.data_in_frac_width
        # )

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "IN_FRAC_WIDTH": self.data_in_frac_width,
            "IN_PARALLELISM": self.in_rows,
            "IN_SIZE": self.in_columns,
            "MAX_LARGE_NUMBERS": self.max_large_numbers,
            "LARGE_NUMBER_THRES": self.large_number_thres,
        }

    def sw_compute(self):
        # for small_large_out only
        ref = []
        for i in range(len(self.data_in.data)):
            current_vector_small = [0] * len(self.data_in.data[0])
            count = 0
            for j in range(
                len(current_vector_small) - 1, -1, -1
            ):  # counting from N down to 0
                entry = self.data_in.data[i][j]
                entry_int = entry >> self.data_in_frac_width
                if (
                    self.sw_large_number_checker(entry_int, thres=127)
                    and count < self.max_large_numbers
                ):
                    # entries with large numbers are masked
                    current_vector_small[j] = 0
                    count += 1
                else:
                    current_vector_small[j] = entry
            ref.append(current_vector_small)
        ref.reverse()
        return ref

    def sw_large_number_checker(self, data, thres):
        # MSB checker for fixed-point 16
        # data is a signed integer
        assert thres > 0, "Large number threshold must be positive!"
        return abs(data) > thres
        # if (data > 0):
        #     return (data >= (2**pos))
        # else:
        #     return (abs(data) >= (2**pos + 1))


def debug_state(dut, state):
    logger.debug(
        "{} State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def cocotb_test_scatter(dut):
    """Test integer based vector mult"""
    samples = 10
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
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out_small.value
        )
        # breakpoint()
        debug_state(dut, "Pre-clk")
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results_signed(test_case.outputs.data, test_case.ref)


@pytest.mark.dev
def test_scatter():
    tb = VerificationCase()
    mase_runner(
        module_param_list=[tb.get_dut_parameters()],
        extra_build_args=["--unroll-count", "3000"],
    )


if __name__ == "__main__":
    test_scatter()
