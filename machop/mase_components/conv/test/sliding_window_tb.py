#!/usr/bin/env python3

import os, logging

from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.runner import mase_runner

import torch

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from math import ceil

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=2):
        self.samples = samples
        self.data_width = 32

        self.kernel_width = 3
        self.kernel_height = 3
        self.img_width = 3
        self.img_height = 4
        self.padding_width = 1
        self.padding_height = 4
        self.stride = 2
        self.channels = 2

        print(
            "parameters:\n\
        kernel_width,kernel_height  = {} by {} \n\
        img_width,img_height        = {} by {} \n\
        padding_width,padding_height= {} by {} \n\
        stride    = {}\n\
        channels  = {}".format(
                self.kernel_width,
                self.kernel_height,
                self.img_width,
                self.img_height,
                self.padding_width,
                self.padding_height,
                self.stride,
                self.channels,
            )
        )
        data = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        data.reverse()
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.img_width * self.img_height * self.channels,
            num=1,
            max_stalls=2 * samples * self.img_width * self.img_height * self.channels,
            is_data_vector=False,
            # data_specify = data,
            debug=debug,
        )

        ## remain modification
        self.out_height = ceil(
            (self.img_height - self.kernel_height + 2 * self.padding_height + 1)
            / self.stride
        )
        self.out_width = ceil(
            (self.img_width - self.kernel_width + 2 * self.padding_width + 1)
            / self.stride
        )
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
            "KERNEL_WIDTH": self.kernel_width,
            "KERNEL_HEIGHT": self.kernel_height,
            "PADDING_WIDTH": self.padding_width,
            "PADDING_HEIGHT": self.padding_height,
            "STRIDE": self.stride,
            "CHANNELS": self.channels,
        }

    def data_pack(self):
        in_width = self.img_width
        in_height = self.img_height
        in_channels = self.channels
        kernel_width = self.kernel_width
        kernel_height = self.kernel_height
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
                        re_data_tensor[i][j][k + padding_height][
                            s + padding_width
                        ] = data_tensor[i][k][s][j]

        return re_data_tensor

    def sw_compute(self):
        data = self.data_pack().tolist()
        ref = []
        for i in range(self.samples):
            for k in range(self.out_height):
                for t in range(self.out_width):
                    for j in range(self.channels):
                        l = [
                            [
                                data[i][j][k * self.stride + s][t * self.stride + m]
                                for m in range(self.kernel_width)
                            ]
                            for s in range(self.kernel_height)
                        ]
                        tensor_l = torch.tensor(l).reshape(-1)
                        ref.append(tensor_l.flip(0).tolist())
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
async def test_sliding_window(dut):
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
    dut.data_out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 200):
        await FallingEdge(dut.clk)
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

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of output:\n\
            {},{} data_in = {}\n\
            {},{} data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            int(dut.data_in.value),
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )
    logger.debug(
        "wave of sws:\n\
        stride_enable = {} \n\
        out_x = {} \n\
        out_y = {} \n\
        out_c = {} \n\
        x_change = {} \n\
        y_change = {} \n\
        count_stride_x = {} \n\
        count_stride_y = {} \n\
        {},{} data_out = {}\n\
            ".format(
            dut.sws_inst.stride_enable.value,
            dut.sws_inst.buffer_x.value,
            dut.sws_inst.buffer_y.value,
            dut.sws_inst.buffer_c.value,
            dut.sws_inst.x_change.value,
            dut.sws_inst.y_change.value,
            dut.sws_inst.count_stride_x.value,
            dut.sws_inst.count_stride_y.value,
            dut.sws_inst.data_out_valid.value,
            dut.sws_inst.data_out_ready.value,
            [int(i) for i in dut.sws_inst.data_out.value],
        )
    )


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
