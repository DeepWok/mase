#!/usr/bin/env python3

import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# from torchsummary import summary
from einops import rearrange, reduce, repeat


import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner


debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=2):
        self.samples = samples
        self.data_width = 32

        self.num = 12
        self.roll_num = 2

        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.num,
            max_stalls=2 * samples,
            is_data_vector=True,
            debug=debug,
        )

        ## remain modification
        ### no padding
        # self.outputs = RandomSink(
        #     samples=samples * self.channels * (
        #         self.img_width - self.kernel_width + 1) * (
        #         self.img_height - self.kernel_height + 1),
        #     max_stalls=2 * samples,
        #     debug=debug,
        # )
        ### padding
        self.outputs = RandomSink(
            samples=samples * self.num / self.roll_num,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_width,
            "NUM": self.num,
            "ROLL_NUM": self.roll_num,
        }

    def sw_compute(self):
        ref = []
        return ref


def debug_state(dut, state):
    logger.debug(
        "{} State: (data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_roller(dut):
    """Test integer based vector mult"""
    samples = 3
    test_case = VerificationCase(samples=samples)

    ## rearrange input to display

    logger.debug(
        "initial data:\n\
    data_in = {}\n".format(
            test_case.data_in.data,
        )
    )
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
    for i in range(samples * 50):
        await FallingEdge(dut.clk)
        dut.data_in_valid.value = test_case.data_in.pre_compute(dut.data_in_ready.value)
        await Timer(1, units="ns")
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        wave_check(dut)
        breakpoint()
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of output:\n\
            data_in = {}\n\
            data_in vr = {},{}\n\
            data_out = {}\n\
            data_out vr = {},{}\n\
            ".format(
            [int(i) for i in dut.data_in.value],
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_out.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
        )
    )
    logger.debug(
        "wave of shift_register:\n\
            shift_register = {}\n\
            ".format(
            [int(i) for i in dut.shift_reg.value]
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/conv/roller.sv",
        "../../../../hardware/common/register_slice.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(
        verilog_sources=verilog_sources,
        toplevel="roller",
        extra_args=extra_args,
    )

    runner.test(toplevel="roller", py_module="roller_tb")


if __name__ == "__main__":
    runner()
