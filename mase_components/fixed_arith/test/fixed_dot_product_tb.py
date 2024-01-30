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
        self.data_in_width = 32
        self.weight_width = 16
        self.vector_size = 4
        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.vector_size,
            max_stalls=0,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples,
            num=self.vector_size,
            max_stalls=0,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "WEIGHT_WIDTH": self.weight_width,
            "IN_SIZE": self.vector_size,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            s = [
                self.data_in.data[i][j] * self.weight.data[i][j]
                for j in range(self.vector_size)
            ]
            ref.append(sum(s))
        ref.reverse()
        return ref


def in_out_wave(dut, name):
    logger.debug(
        "{}  State: (w_valid, w_ready, in_valid,in_ready,out_valid,out_ready) = ({},{},{},{})".format(
            name,
            dut.weight_valid.value,
            dut.weight_ready.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_dot_product(dut):
    """Test integer based vector mult"""
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
    dut.weight_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")
        ## pre_compute
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        ## compute
        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        in_out_wave(dut, "Pre-clk")

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
