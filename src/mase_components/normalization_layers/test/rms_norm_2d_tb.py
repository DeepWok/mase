#!/usr/bin/env python3

import logging
from random import randint
from os import makedirs
from pathlib import Path

# from itertools import batched  # Python 3.12
from itertools import repeat
from math import ceil, log2

import torch
from torch import nn
import numpy as np
import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    ErrorThresholdStreamMonitor,
)
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)
from mase_cocotb.utils import (
    bit_driver,
    batched,
    sign_extend_t,
    verilator_str_param,
)

from chop.nn.quantized.modules import RMSNormInteger
from chop.nn.quantizers.quantizers_for_hw import integer_floor_quantizer_for_hw


from mase_components.scalar_operators.fixed.test.isqrt_sw import make_lut
from mase_components.common.test.lut_tb import write_memb

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class RMSNorm2dTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "TOTAL_DIM0",
                "TOTAL_DIM1",
                "COMPUTE_DIM0",
                "COMPUTE_DIM1",
                "CHANNELS",
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "SCALE_WIDTH",
                "SCALE_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                "ISQRT_WIDTH",
                "ISQRT_FRAC_WIDTH",
                "DEPTH_DIM0",
                "DEPTH_DIM1",
                "NUM_VALUES",
            ]
        )

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Model
        self.quantized_model = RMSNormInteger(
            normalized_shape=[self.CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0],
            elementwise_affine=True,
            config={
                "data_in_width": self.IN_WIDTH,
                "data_in_frac_width": self.IN_FRAC_WIDTH,
                "weight_width": self.SCALE_WIDTH,
                "weight_frac_width": self.SCALE_FRAC_WIDTH,
            },
        )
        self.quantized_model.eval()

        # Drivers & Monitors
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight_data, dut.weight_valid, dut.weight_ready
        )

        # Bit Error calculation
        error_bits = 4

        # If we want the output frac to have larger width, we can expect a
        # larger rounding error difference between the integer and float models
        if self.OUT_FRAC_WIDTH > self.IN_FRAC_WIDTH:
            error_bits += 2 ** (self.OUT_FRAC_WIDTH - self.IN_FRAC_WIDTH)

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            name="Output Monitor",
            width=self.OUT_WIDTH,
            signed=True,
            error_bits=error_bits,
        )

    def generate_inputs(self, num=2):
        # Input Data
        inputs = list()
        for _ in range(self.CHANNELS * num):
            inputs.extend(
                gen_random_matrix_input(
                    *self.total_tup, *self.compute_tup, *self.in_width_tup
                )
            )

        # Model weights (scale)
        weights = list()
        for _ in range(self.CHANNELS):
            weights.extend(
                gen_random_matrix_input(
                    *self.total_tup, *self.compute_tup, *self.in_width_tup
                )
            )

        # Set weight tensor into model
        weights_t = self.reconstruct_tensor(
            weights, self.SCALE_WIDTH, self.SCALE_FRAC_WIDTH
        )
        self.quantized_model.weight = nn.Parameter(weights_t)

        # Weights are same across all batches, however we need to repeat them to the driver
        repeated_weights = list()
        for _ in range(num):
            repeated_weights.extend(weights)
        return inputs, repeated_weights

    def reconstruct_tensor(self, x, width, frac_width):
        batches = batched(x, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [
            rebuild_matrix(b, *self.total_tup, *self.compute_tup) for b in batches
        ]
        x = torch.stack(matrix_list).reshape(
            -1, self.CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, width).to(dtype=torch.float32) / (2**frac_width)
        return x

    def output_monitor_split(self, x, width, frac_width):
        x = integer_quantizer_for_hw(x, width, frac_width)
        x = x.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(x.shape[0]):
            model_out.extend(split_matrix(x[i], *self.total_tup, *self.compute_tup))
        return model_out

    def model(self, inputs):
        # Input reconstruction
        x = self.reconstruct_tensor(inputs, self.IN_WIDTH, self.IN_FRAC_WIDTH)

        # Quantized Software model
        float_y = self.quantized_model(x)

        # Output beat reconstruction
        y = self.output_monitor_split(float_y, self.OUT_WIDTH, self.OUT_FRAC_WIDTH)
        return y

    async def run_test(self, num=2, us=100):
        inputs, weights = self.generate_inputs(num=num)
        self.in_driver.load_driver(inputs)
        self.weight_driver.load_driver(weights)
        exp_out = self.model(inputs)
        self.output_monitor.load_monitor(exp_out)
        await Timer(us, "us")
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    await tb.run_test(num=2, us=10)


@cocotb.test()
async def stream(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    await tb.run_test(num=1000, us=400)

    # Error analysis
    # import json
    # errs = np.stack(tb.output_monitor.error_log).flatten()
    # logger.info("Mean bit-error: %s" % errs.mean())
    # jsonfile = Path(__file__).parent / "data" / f"rms-{tb.IN_WIDTH}.json"
    # with open(jsonfile, 'w') as f:
    #     json.dump({
    #         "mean": errs.mean().item(),
    #         "error": errs.tolist(),
    #     }, f, indent=4)


@cocotb.test()
async def backpressure(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.run_test(num=100, us=100)


@cocotb.test()
async def valid_toggle(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(num=100, us=100)


@cocotb.test()
async def valid_backpressure(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.run_test(num=100, us=200)


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_rms_norm_2d():
    # Consts
    LUT_POW = 7

    mem_dir = Path(__file__).parent / "build" / "group_norm_2d" / "mem"
    makedirs(mem_dir, exist_ok=True)

    def isqrt_width(
        total_dim0,
        total_dim1,
        compute_dim0,
        compute_dim1,
        group_channels,
        in_width,
    ):
        depth_dim0 = total_dim0 // compute_dim0
        depth_dim1 = total_dim1 // compute_dim1
        num_iters = depth_dim0 * depth_dim1 * group_channels
        iter_width = ceil(log2(num_iters))
        square_width = in_width * 2
        adder_width = ceil(log2(compute_dim0 * compute_dim1)) + square_width
        acc_width = adder_width + iter_width
        print(" --- CALCULATED ISQRT WIDTH:", acc_width)
        return acc_width

    def gen_cfg(
        total_dim0: int = 4,
        total_dim1: int = 4,
        compute_dim0: int = 2,
        compute_dim1: int = 2,
        channels: int = 2,
        in_width: int = 8,
        in_frac_width: int = 4,
        scale_width: int = 8,
        scale_frac_width: int = 4,
        out_width: int = 8,
        out_frac_width: int = 4,
        str_id: str = "default",
    ):
        isqrt_w = isqrt_width(
            total_dim0, total_dim1, compute_dim0, compute_dim1, channels, in_width
        )
        lut = make_lut(2**LUT_POW, isqrt_w)
        mem_path = mem_dir / f"lutmem-{str_id}.mem"
        write_memb(mem_path, lut, isqrt_w)
        params = {
            "TOTAL_DIM0": total_dim0,
            "TOTAL_DIM1": total_dim1,
            "COMPUTE_DIM0": compute_dim0,
            "COMPUTE_DIM1": compute_dim1,
            "CHANNELS": channels,
            "IN_WIDTH": in_width,
            "IN_FRAC_WIDTH": in_frac_width,
            "SCALE_WIDTH": scale_width,
            "SCALE_FRAC_WIDTH": scale_frac_width,
            "OUT_WIDTH": out_width,
            "OUT_FRAC_WIDTH": out_frac_width,
            "ISQRT_LUT_MEMFILE": verilator_str_param(str(mem_path)),
            "ISQRT_LUT_POW": LUT_POW,
        }
        return params

    mase_runner(
        module_param_list=[
            gen_cfg(),
            # Rectangle
            gen_cfg(4, 6, 2, 2, 2, 8, 4, 8, 4, 8, 4, "rect0"),
            gen_cfg(6, 2, 2, 2, 2, 8, 4, 8, 4, 8, 4, "rect1"),
            gen_cfg(6, 2, 3, 2, 2, 8, 4, 8, 4, 8, 4, "rect2"),
            gen_cfg(4, 6, 2, 3, 2, 8, 4, 8, 4, 8, 4, "rect3"),
            # Channels
            gen_cfg(4, 4, 2, 2, 1, 8, 4, 8, 4, 8, 4, "channels0"),
            gen_cfg(4, 4, 2, 2, 3, 8, 4, 8, 4, 8, 4, "channels1"),
            # Precision
            gen_cfg(4, 4, 2, 2, 2, 8, 4, 8, 4, 8, 2, "down_frac"),
            gen_cfg(4, 4, 2, 2, 2, 8, 4, 8, 4, 8, 6, "up_frac"),
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_rms_norm_2d()
