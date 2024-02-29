#!/usr/bin/env python3

import logging
from random import randint
# from itertools import batched  # Python 3.12

import torch
from torch import Tensor
import torch.nn.functional as F
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
    sign_extend_t,
    signed_to_unsigned,
    int_floor_quantizer,
)

from mase_components.cast.test.fixed_signed_cast_tb import _fixed_signed_cast_model
from chop.passes.graph.transforms.quantize.quantizers.quantizers_for_hw import (
    integer_quantizer_for_hw,
)
from chop.passes.graph.transforms.quantize.quantizers.integer import (
    integer_quantizer,
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

# Apparently this function only exists in Python 3.12 ...
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class GroupNorm2dTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "GROUP_CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "VARIANCE_WIDTH", "VARIANCE_FRAC_WIDTH",
            "INV_SQRT_WIDTH", "INV_SQRT_FRAC_WIDTH",
            "DEPTH_DIM0", "DEPTH_DIM1",
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

        self.in_driver.log.setLevel(logging.DEBUG)
        self.output_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, num=2):
        inputs = list()
        for _ in range(self.GROUP_CHANNELS * num):
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
            -1, self.GROUP_CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (2 ** self.IN_FRAC_WIDTH)
        logger.debug("Input:")
        logger.debug(x[0])

        # Mean calculation
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        logger.debug("Mu:")
        logger.debug(mu[0])
        mu = int_floor_quantizer(mu, self.IN_WIDTH, self.IN_FRAC_WIDTH)
        logger.debug("Mu:")
        logger.debug(mu[0])

        # Variance calculation
        var = ((x - mu) ** 2).mean(dim=(1, 2, 3), keepdim=True)
        var = int_floor_quantizer(var, self.VARIANCE_WIDTH, self.VARIANCE_FRAC_WIDTH)
        logger.debug("Variance:")
        logger.debug(var[0])

        # Inverse Square Root calculation
        # inv_sqrt = inv_sqrt_model(var)  # TODO: Add inv sqrt model
        inv_sqrt = torch.full_like(var, 0.25)  # TODO: remove this later
        inv_sqrt = int_floor_quantizer(inv_sqrt, self.INV_SQRT_WIDTH, self.INV_SQRT_FRAC_WIDTH)
        logger.debug("Inverse SQRT:")
        logger.debug(inv_sqrt[0])

        # Norm calculation
        diff = x - mu
        norm_out = diff * inv_sqrt
        _, norm_out = _fixed_signed_cast_model(
            norm_out, self.OUT_WIDTH, self.OUT_FRAC_WIDTH,
            symmetric=False, rounding_mode="floor"
        )
        logger.debug("Diff:")
        logger.debug(diff[0])
        logger.debug("Norm:")
        logger.debug(norm_out[0])

        # Rescale & Reshape for output monitor
        norm_out *= (2 ** self.OUT_FRAC_WIDTH)
        logger.debug("Norm (signed):")
        logger.debug(norm_out[0])
        norm_out = signed_to_unsigned(norm_out.to(dtype=torch.int32), self.OUT_WIDTH)
        logger.debug("Norm (unsigned):")
        logger.debug(norm_out[0])
        y = norm_out.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)

        # Output beat reconstruction
        model_out = list()
        for i in range(y.shape[0]):
            model_out.extend(split_matrix(y[i], *self.total_tup, *self.compute_tup))
        return model_out


@cocotb.test()
async def basic(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs()
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(10, 'us')
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(seed=0, trace=True)
