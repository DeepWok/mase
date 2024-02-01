#!/usr/bin/env python3

# This script tests the fix-point vector mult
import os, logging

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
        self.data_in_width = 32
        self.weight_width = 16
        self.vector_size = 2
        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sweight_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "WEIGHT_WIDTH": self.weight_width,
            "IN_SIZE": self.vector_size,
        }

    def sweight_compute(self):
        ref = []
        for i in range(self.samples):
            s = [
                self.data_in.data[i][j] * self.weight.data[i][j]
                for j in range(self.vector_size)
            ]
            ref.append(s)
        ref.reverse()
        return ref


# Check if an is_impossible_ state is reached
def is_impossible_state(
    weight_ready,
    weight_valid,
    data_in_ready,
    data_in_valid,
    data_out_ready,
    data_out_valid,
):
    return False


@cocotb.test()
async def test_fixed_vector_mult(dut):
    """Test integer based vector mult"""
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
    dut.weight_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    logger.debug(
        "Pre-clk  State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    await FallingEdge(dut.clk)
    logger.debug(
        "Post-clk State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    logger.debug(
        "Pre-clk  State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )
    await FallingEdge(dut.clk)
    logger.debug(
        "Post-clk State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        logger.debug(
            "Post-clk State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
                dut.weight_ready.value,
                dut.weight_valid.value,
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )
        assert not is_impossible_state(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        ), "Error: invalid state (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )

        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        logger.debug(
            "Pre-clk State0: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
                dut.weight_ready.value,
                dut.weight_valid.value,
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )
        await Timer(1, units="ns")
        logger.debug(
            "Pre-clk State1: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
                dut.weight_ready.value,
                dut.weight_valid.value,
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        logger.debug(
            "Pre-clk State2: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
                dut.weight_ready.value,
                dut.weight_valid.value,
                dut.data_in_ready.value,
                dut.data_in_valid.value,
                dut.data_out_ready.value,
                dut.data_out_valid.value,
            )
        )

        if (
            test_case.weight.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
