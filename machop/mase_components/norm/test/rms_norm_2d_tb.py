#!/usr/bin/env python3

import logging
from random import randint
# from itertools import batched  # Python 3.12

import torch
from torch import Tensor
import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
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
)

from chop.passes.graph.transforms.quantize.quantizers.integer import (
    integer_floor_quantizer
)

from mase_components.cast.test.fixed_signed_cast_tb import (
    _fixed_signed_cast_model
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def _fixed_rms_norm_2d_model(
    x: Tensor,
    acc_width: int,
    acc_frac_width: int,
    inv_sqrt_width: int,
    inv_sqrt_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Sum of Squares
    sum_sq = torch.square(x).sum(dim=(1, 2, 3), keepdim=True)
    sum_sq = integer_floor_quantizer(sum_sq, acc_width, acc_frac_width)
    logger.debug("Sum of Squares:")
    logger.debug(sum_sq[0])

    # Divide to get mean square
    mean_sq = sum_sq / (x.shape[1] * x.shape[2] * x.shape[3])
    mean_sq = integer_floor_quantizer(mean_sq, acc_width, acc_frac_width)
    logger.debug("Mean Square:")
    logger.debug(mean_sq[0])

    # Get inverse sqrt of mean square
    # inv_sqrt = inv_sqrt_model(mean_sq)  # TODO: Add inv sqrt model
    inv_sqrt = torch.full_like(mean_sq, 0.25)  # TODO: remove this later
    inv_sqrt = integer_floor_quantizer(inv_sqrt, inv_sqrt_width, inv_sqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(inv_sqrt[0])

    # Norm calculation
    norm_out = x * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out[0])
    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width,
        symmetric=False, rounding_mode="floor"
    )

    return norm_out_float, norm_int_out


class RMSNorm2dTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "INV_SQRT_WIDTH", "INV_SQRT_FRAC_WIDTH",
            "ACC_WIDTH", "ACC_FRAC_WIDTH",
            "DEPTH_DIM0", "DEPTH_DIM1",
            "NUM_VALUES"
        ])

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready
        )

    def generate_inputs(self, num=2):
        inputs = list()
        for _ in range(self.CHANNELS * num):
            inputs.extend(gen_random_matrix_input(
                *self.total_tup, *self.compute_tup, *self.in_width_tup
            ))
        return inputs

    def model(self, inputs):
        # Input reconstruction
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list).reshape(
            -1, self.CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (2 ** self.IN_FRAC_WIDTH)

        # Float model
        norm_out_float, norm_int_out = _fixed_rms_norm_2d_model(
            x=x,
            acc_width=self.ACC_WIDTH,
            acc_frac_width=self.ACC_FRAC_WIDTH,
            inv_sqrt_width=self.INV_SQRT_WIDTH,
            inv_sqrt_frac_width=self.INV_SQRT_FRAC_WIDTH,
            out_width=self.OUT_WIDTH,
            out_frac_width=self.OUT_FRAC_WIDTH,
        )

        # Output beat reconstruction
        y = norm_int_out.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(y.shape[0]):
            model_out.extend(split_matrix(y[i], *self.total_tup, *self.compute_tup))
        return model_out


@cocotb.test()
async def basic(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=2)
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(10, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def stream(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=600)
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(200, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def backpressure(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))

    inputs = tb.generate_inputs(num=100)
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(100, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_toggle(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)

    inputs = tb.generate_inputs(num=100)
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(100, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_backpressure(dut):
    tb = RMSNorm2dTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))

    inputs = tb.generate_inputs(num=100)
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(200, 'us')
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner()
