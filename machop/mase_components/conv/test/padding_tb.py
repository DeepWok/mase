#!/usr/bin/env python3

import random, os, math, logging, sys

from mase_cocotb.random_test import RandomSource, RandomSink, check_results

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
from math import ceil

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_width = 32

        self.padding_height = 1
        self.padding_width = 3
        self.img_height = 2
        self.img_width = 3
        self.channels = 1

        # data = [0,1,2,3,4,5,6,7]
        # data.reverse()
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.img_width * self.img_height * self.channels,
            num=1,
            max_stalls=2 * samples * self.img_width * self.img_height * self.channels,
            is_data_vector=False,
            # Data_Specify = data,
            debug=debug,
        )

        ## remain modification
        ## no padding
        self.out_height = self.img_height + 2 * self.padding_height
        self.out_width = self.img_width + 2 * self.padding_width
        self.outputs = RandomSink(
            samples=samples * self.channels * self.out_height * self.out_width,
            max_stalls=2 * samples * self.channels * self.out_height * self.out_width,
            debug=debug,
        )

        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_width,
            "IMG_WIDTH": self.img_width,
            "IMG_HEIGHT": self.img_height,
            "PADDING_HEIGHT": self.padding_height,
            "PADDING_WIDTH": self.padding_width,
            "CHANNELS": self.channels,
        }

    def data_pack(self):
        in_width = self.img_width
        in_height = self.img_height
        in_channels = self.channels
        padding_width = self.padding_width
        padding_height = self.padding_height
        samples = self.samples
        # data_pack
        data_tensor = torch.tensor(self.data_in.data, dtype=torch.float)
        data_tensor = data_tensor.flip(0).reshape(
            samples, in_height, in_width, in_channels
        )
        re_data_tensor = torch.zeros(
            samples,
            in_channels,
            in_height + 2 * padding_height,
            in_width + 2 * padding_width,
        )
        for i in range(samples):
            for j in range(in_channels):
                for k in range(in_height):
                    for s in range(in_width):
                        re_data_tensor[i][j][k + padding_height][s + padding_width] = (
                            data_tensor[i][k][s][j]
                        )

        return re_data_tensor

    def sw_compute(self):
        out_width = self.img_width + 2 * self.padding_width
        out_height = self.img_height + 2 * self.padding_height
        in_channels = self.channels
        samples = self.samples
        data = self.data_pack()
        out_data = torch.zeros(samples, out_height, out_width, in_channels)
        for i in range(samples):
            for j in range(in_channels):
                for k in range(out_height):
                    for s in range(out_width):
                        out_data[i][k][s][j] = data[i][j][k][s]
        ref = [int(i) for i in out_data.reshape(-1).tolist()]
        # ref.reverse()
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
async def test_padding(dut):
    """Test integer based vector mult"""
    samples = 20
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
    dut.data_out_ready.value = 0
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 150):
        await FallingEdge(dut.clk)
        # pre_compute
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        # wave_check(dut)
        # breakpoint()
        if test_case.data_in.is_empty() and test_case.outputs.is_full():
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    print([int(i) for i in test_case.outputs.data])
    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of output:\n\
            {},{} data_in = {}\n\
            {},{} register_out = {}\n\
            {},{} data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            int(dut.data_in.value),
            dut.register_out_valid.value,
            dut.register_out_ready.value,
            int(dut.register_out.value),
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            int(dut.data_out.value),
        )
    )
    logger.debug(
        "wave of inner:\n\
            padding = {}\n\
            count_x = {}\n\
            count_y = {}\n\
            count_c = {}\n\
            ".format(
            dut.padding.value,
            int(dut.count_x.value),
            int(dut.count_y.value),
            int(dut.count_c.value),
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../mase_components/conv/rtl/padding.sv",
        "../../../mase_components/common/rtl/register_slice.sv",
        "../../../mase_components/common/rtl/skid_buffer.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="padding",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="padding", test_module="padding_tb")


if __name__ == "__main__":
    # for i in range(20):
    runner()
