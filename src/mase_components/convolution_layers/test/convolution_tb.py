#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge
from cocotb.decorators import coroutine

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    ErrorThresholdStreamMonitor,
)
from mase_cocotb.runner import mase_runner
from math import ceil

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized.modules.conv2d import Conv2dInteger
from chop.nn.quantizers import integer_quantizer
from mase_cocotb.z_qlayers import quantize_to_int as q2i

import torch.nn.functional as F


class ConvArithTB(Testbench):
    def __init__(self, dut, samples=1) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.samples = samples

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )

        self.bias_driver = StreamDriver(
            dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        )

        self.data_out_0_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            width=self.get_parameter("DATA_OUT_0_PRECISION_0"),
            signed=True,
            error_bits=1,
            check=True,
        )
        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)
        self.bias_driver.log.setLevel(logging.DEBUG)
        torch.manual_seed(0)
        self.model = Conv2dInteger(
            in_channels=self.get_parameter("IN_C"),
            out_channels=self.get_parameter("OUT_C"),
            kernel_size=(
                self.get_parameter("KERNEL_Y"),
                self.get_parameter("KERNEL_X"),
            ),
            stride=self.get_parameter("STRIDE"),
            padding=(self.get_parameter("PADDING_Y"), self.get_parameter("PADDING_X")),
            bias=True if self.get_parameter("HAS_BIAS") == 1 else False,
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "weight_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "bias_width": self.get_parameter("BIAS_PRECISION_0"),
                "bias_frac_width": self.get_parameter("BIAS_PRECISION_1"),
                "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "data_out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
        )

    def generate_data(self, config):
        torch.manual_seed(0)

        def get_manual_result(
            x,
            w,
            b,
            stride,
            samples,
            kernel_y,
            kernel_x,
            in_y,
            in_x,
            padding_y,
            padding_x,
            roll_in_num,
            oc,
        ):
            input_unf = F.unfold(x, (kernel_x, kernel_y), stride=stride)
            input_unf = input_unf.view(samples, ic, kernel_x, kernel_y, -1)
            input_unf = input_unf.permute(0, 4, 1, 2, 3)  # (N, L, C, KH, KW)
            out_height = ceil((in_y - kernel_y + 2 * padding_y + 1) / stride)
            out_width = ceil((in_x - kernel_x + 2 * padding_x + 1) / stride)

            sliding_num = out_width * out_height

            qx = input_unf.reshape(samples, sliding_num, roll_in_num)
            qw = w.reshape(1, 1, oc, roll_in_num).repeat(samples, sliding_num, 1, 1)
            qb = b.reshape(1, 1, oc).repeat(samples, sliding_num, 1)
            out2 = torch.sum(qx.unsqueeze(2) * qw, dim=-1) + qb
            return out2

        samples = self.samples
        ic = self.get_parameter("IN_C")
        iy = self.get_parameter("IN_Y")
        ix = self.get_parameter("IN_X")
        # get parameters with integer format
        x = 5 * torch.randn(samples, ic, iy, ix)

        _dict = self.model.get_quantized_weights_with_inputs(x)
        x, w, b, out = _dict["x"], _dict["w"], _dict["bias"], _dict["y"]
        # out2 = get_manual_result(x, w, b, 2,1,2,2,4,4,0,0,12,4)
        # data_in_pack

        x = q2i(
            x,
            config["data_in_width"],
            config["data_in_frac_width"],
        )

        self.log.info(f"x = {x}")
        # from (samples, c, h, w) to (samples*h*w*c/unroll_in_c, unroll_in_c)
        unroll_ic = self.get_parameter("UNROLL_IN_C")
        hw_x = (x.permute(0, 2, 3, 1).reshape(-1).reshape(-1, unroll_ic)).tolist()
        # parameters packs

        self.log.info(f"weight = {w}")
        self.log.info(f"bias = {b}")
        w = q2i(
            w,
            config["weight_width"],
            config["weight_frac_width"],
        )
        b = q2i(
            b,
            config["bias_width"],
            config["bias_frac_width"],
        )
        self.log.info(f"weight = {w}")
        self.log.info(f"bias = {b}")
        hw_w, hw_b = self.conv_pack(
            weight=w,
            bias=b,
            in_channels=ic,
            kernel_size=[
                self.get_parameter("KERNEL_X"),
                self.get_parameter("KERNEL_Y"),
            ],
            out_channels=self.get_parameter("OUT_C"),
            unroll_in_channels=self.get_parameter("UNROLL_IN_C"),
            unroll_kernel_out=self.get_parameter("UNROLL_KERNEL_OUT"),
            unroll_out_channels=self.get_parameter("UNROLL_OUT_C"),
        )
        exp_out = q2i(
            out,
            config["out_width"],
            config["out_frac_width"],
        )
        exp_out = (
            exp_out.reshape(
                -1, self.get_parameter("OUT_C"), self.get_parameter("SLIDING_NUM")
            )
            .permute(0, 2, 1)
            .reshape(-1, self.get_parameter("UNROLL_OUT_C"))
            .tolist()
        )
        self.log.info(f"Processing outs: {exp_out}")
        return hw_x, hw_w, hw_b, exp_out

    def conv_pack(
        self,
        weight,
        bias,
        in_channels,
        kernel_size,
        out_channels,
        unroll_in_channels,
        unroll_kernel_out,
        unroll_out_channels,
    ):
        samples = self.samples * self.get_parameter("SLIDING_NUM")
        # requires input as a quantized int format
        # weight_pack
        # from (oc,ic/u_ic,u_ic,h,w) to (ic/u_ic,h*w,u_ic,oc)
        reorder_w_tensor = (
            weight.repeat(samples, 1, 1, 1, 1)
            .reshape(
                samples,
                out_channels,
                int(in_channels / unroll_in_channels),
                unroll_in_channels,
                kernel_size[0] * kernel_size[1],
            )
            .permute(0, 2, 4, 3, 1)
        )

        # reverse the final 2 dimension
        # from(samples, int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_kernel_out, int(out_channels/unroll_out_channels), unroll_out_channels)
        # to  (samples, int(out_channels/unroll_out_channels), int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_out_channels, unroll_kernel_out)
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_size[0] * kernel_size[1] * in_channels / unroll_kernel_out),
            unroll_kernel_out,
            int(out_channels / unroll_out_channels),
            unroll_out_channels,
        ).permute(0, 3, 1, 4, 2)

        w_tensor = w_tensor.reshape(
            -1,
            unroll_out_channels * unroll_kernel_out,
        )
        w_in = w_tensor.type(torch.int).tolist()
        # bias_pack
        bias_tensor = (
            bias.repeat(samples, 1).reshape(-1).reshape(-1, unroll_out_channels)
        )
        b_in = bias_tensor.type(torch.int).tolist()
        return w_in, b_in

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        x, w, b, o = self.generate_data(
            {
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "weight_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "bias_width": self.get_parameter("BIAS_PRECISION_0"),
                "bias_frac_width": self.get_parameter("BIAS_PRECISION_1"),
                "out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            }
        )
        # * Load the inputs driver
        self.log.info(f"Processing inputs: {x}")
        self.data_in_0_driver.load_driver(x)

        # * Load the weights driver
        self.log.info(f"Processing weights: {w}")
        self.weight_driver.load_driver(w)

        # * Load the bias driver
        self.log.info(f"Processing bias: {b}")
        self.bias_driver.load_driver(b)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {o}")
        self.data_out_0_monitor.load_monitor(o)

        # cocotb.start_soon(check_signal(self.dut, self.log))
        await Timer(100, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = ConvArithTB(dut, 10)
    await tb.run_test()


async def check_signal(dut, log):
    while True:
        await RisingEdge(dut.clk)
        handshake_signal_check(dut.kernel_valid, dut.kernel_ready, dut.kernel, log)
        # handshake_signal_check(dut.rolled_k_valid, dut.rolled_k_ready, dut.rolled_k, log)
        # handshake_signal_check(dut.bias_valid,
        #                        dut.bias_ready,
        #                        dut.bias, log)


def handshake_signal_check(valid, ready, signal, log):
    svalue = [i.signed_integer for i in signal.value]
    if valid.value & ready.value:
        log.debug(f"handshake {signal} = {svalue}")


def get_fixed_conv_config(kwargs={}):
    config = {
        "IN_C": 3,
        "UNROLL_IN_C": 3,
        "IN_X": 3,
        "IN_Y": 3,
        "KERNEL_X": 3,
        "KERNEL_Y": 2,
        "UNROLL_KERNEL_OUT": 3,
        "OUT_C": 4,
        "UNROLL_OUT_C": 2,
        "STRIDE": 2,
        "PADDING_Y": 1,
        "PADDING_X": 2,
        "HAS_BIAS": 1,
    }
    in_y = config["IN_Y"]
    in_x = config["IN_X"]
    kernel_y = config["KERNEL_Y"]
    kernel_x = config["KERNEL_X"]
    padding_y = config["PADDING_Y"]
    padding_x = config["PADDING_X"]
    stride = config["STRIDE"]
    out_height = ceil((in_y - kernel_y + 2 * padding_y + 1) / stride)
    out_width = ceil((in_x - kernel_x + 2 * padding_x + 1) / stride)

    sliding_num = out_width * out_height
    config["SLIDING_NUM"] = sliding_num
    config.update(kwargs)
    return config


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_conv_config(),
        ],
    )


def test_fixed_linear_regression():
    """
    More extensive tests to check realistic parameter sizes.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_conv_config(
                {
                    "HAS_BIAS": 1,
                    "ROLL_IN_NUM": 3 * 16 * 16,
                    "ROLL_OUT_NUM": 4,
                    "IN_CHANNELS_DEPTH": 1,
                    "OUT_CHANNELS_PARALLELISM": 4,
                    "WEIGHT_REPEATS": 14 * 14,
                    "OUT_CHANNELS_DEPTH": 96,
                }
            ),
            # get_fixed_linear_config(
            #     {
            #         "HAS_BIAS": 1,
            #         "WEIGHTS_PRE_TRANSPOSED": 0,
            #         "DATA_IN_0_TENSOR_SIZE_DIM_0": 768,
            #         "DATA_IN_0_PARALLELISM_DIM_0": 32,
            #         "WEIGHT_TENSOR_SIZE_DIM_0": 768,
            #         "WEIGHT_TENSOR_SIZE_DIM_1": 768,
            #         "WEIGHT_PARALLELISM_DIM_0": 32,
            #         "WEIGHT_PARALLELISM_DIM_1": 32,
            #         "BIAS_TENSOR_SIZE_DIM_0": 768,
            #         "BIAS_PARALLELISM_DIM_0": 32,
            #     }
            # ),
        ],
    )


if __name__ == "__main__":
    test_fixed_linear_smoke()
    # test_fixed_linear_regression()
