#!/usr/bin/env python3

import logging
from functools import partial
from random import randint
from math import ceil, log2
from copy import copy
from pathlib import Path
from os import makedirs
# from itertools import batched  # Python 3.12

import torch
from torch import Tensor
import numpy as np
import cocotb
from cocotb.result import TestFailure
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor
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

from chop.passes.graph.transforms.quantize.quantized_modules import (
    GroupNormInteger,
)
from chop.passes.graph.transforms.quantize.quantizers.quantizers_for_hw import (
    integer_floor_quantizer_for_hw,
    integer_quantizer_for_hw
)
from chop.passes.graph.transforms.quantize.quantizers.integer import (
    integer_floor_quantizer
)

from mase_components.cast.test.fixed_signed_cast_tb import (
    _fixed_signed_cast_model
)
from mase_components.fixed_arithmetic.test.isqrt_sw import (
    lut_parameter_dict, make_lut, isqrt_sw2
)
from mase_components.common.test.lut_tb import write_memb

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def _fixed_group_norm_2d_model(
    x: Tensor,
    in_width: int,
    in_frac_width: int,
    diff_width: int,
    diff_frac_width: int,
    square_width: int,
    square_frac_width: int,
    variance_width: int,
    variance_frac_width: int,
    isqrt_width: int,
    isqrt_frac_width: int,
    isqrt_lut: list,
    norm_width: int,
    norm_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Mean calculation
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    logger.debug("Mu:")
    logger.debug(mu[0])
    mu = integer_floor_quantizer(mu, in_width, in_frac_width)
    mu_int = integer_floor_quantizer_for_hw(mu.clone(), in_width, in_frac_width)
    logger.debug("Mu Quantized:")
    logger.debug(mu[0])

    # Variance calculation
    diff = x - mu
    diff_int = integer_floor_quantizer_for_hw(diff.clone(), diff_width, diff_frac_width)
    logger.debug("Diff:")
    logger.debug(diff[0])

    squares = diff ** 2
    logger.debug("Squares:")
    logger.debug(squares[0])
    squares_int = (squares * (2**square_frac_width)).int()
    logger.debug(squares * (2**square_frac_width))

    sum_squares = torch.sum(squares, dim=(1, 2, 3), keepdim=True)
    sum_squares = integer_floor_quantizer(sum_squares, variance_width, variance_frac_width)
    sum_squares_int = integer_floor_quantizer_for_hw(sum_squares.clone(), variance_width, variance_frac_width)

    num_vals = x.shape[1] * x.shape[2] * x.shape[3]
    logger.debug("Num Values: %d" % (num_vals))
    var = sum_squares / num_vals
    var = integer_floor_quantizer(var, variance_width, variance_frac_width)
    var_i = integer_floor_quantizer_for_hw(var.clone(), variance_width, variance_frac_width)
    logger.debug("Variance:")
    logger.debug(f"{var[0]}")

    # Clamp down variance to isqrt width
    var_clamp = torch.clamp(var, 0.0, ((2**isqrt_width)-1)/(2**isqrt_frac_width))
    logger.debug("Variance Clamped:")
    logger.debug(f"{var_clamp[0]}")
    var_clamp_int = (var_clamp * (2 ** isqrt_frac_width)).int()

    # Inverse Square Root calculation
    lut_pow = ceil(log2(len(isqrt_lut)))
    logger.debug("Variance INT:")
    logger.debug(f"{var_clamp_int[0]}")

    f = partial(
        isqrt_sw2,
        in_width=isqrt_width,
        frac_width=isqrt_frac_width,
        lut_pow=lut_pow,
        lut=isqrt_lut,
        debug=False,
    )
    inv_sqrt_int = var_clamp_int.clone().apply_(f)

    logger.debug("INV SQRT INT:")
    logger.debug(f"{inv_sqrt_int[0]}")

    inv_sqrt = inv_sqrt_int / (2 ** isqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(f"{inv_sqrt[0]}")

    # Norm calculation
    norm_out = diff * inv_sqrt
    norm_int = integer_floor_quantizer_for_hw(
        norm_out.clone(), norm_width, norm_frac_width
    )
    logger.debug("Norm:")
    logger.debug(norm_out[0])

    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width,
        symmetric=False, rounding_mode="floor"
    )
    logger.debug("Norm (Casted):")
    logger.debug(norm_out_float[0])

    return norm_out_float, norm_int_out, {
        "mu": mu_int.squeeze(dim=(1, 2, 3)).tolist(),
        "squares": squares_int,
        "sum_squares": sum_squares_int.squeeze(dim=(1, 2, 3)).tolist(),
        "var": var_i.squeeze(dim=(1, 2, 3)).tolist(),
        "var_clamp": var_clamp_int.squeeze(dim=(1, 2, 3)).tolist(),
        "isqrt": inv_sqrt_int.squeeze(dim=(1, 2, 3)).tolist(),
        "diff": diff_int,
        "norm": norm_int,
    }


class GroupNorm2dTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "GROUP_CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "DIFF_WIDTH", "DIFF_FRAC_WIDTH",
            "SQUARE_WIDTH", "SQUARE_FRAC_WIDTH",
            "VARIANCE_WIDTH", "VARIANCE_FRAC_WIDTH",
            "ISQRT_WIDTH", "ISQRT_FRAC_WIDTH",
            "NORM_WIDTH", "NORM_FRAC_WIDTH",
            "DEPTH_DIM0", "DEPTH_DIM1",
        ])

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Inverse Square Root LUT
        self.isqrt_lut = make_lut(2**5, 16)

        self.num_groups = randint(2, 3)
        self.total_channels = self.GROUP_CHANNELS * self.num_groups

        self.quantized_model = GroupNormInteger(
            num_groups=self.num_groups,
            num_channels=self.total_channels,
            affine=False,
            config={
                "data_in_width": self.IN_WIDTH,
                "data_in_frac_width": self.IN_FRAC_WIDTH,
            }
        )

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )

        # Bit Error calculation
        error_bits = 2

        # If we want the output frac to have larger width, we can expect a
        # larger rounding error difference between the integer and float models
        if self.OUT_FRAC_WIDTH > self.IN_FRAC_WIDTH:
            error_bits += 2 ** (self.OUT_FRAC_WIDTH - self.IN_FRAC_WIDTH)

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready,
            name="Output Monitor",
            width=self.OUT_WIDTH,
            signed=True,
            error_bits=error_bits,
        )

    def generate_inputs(self, batches=1):
        inputs = list()
        for _ in range(self.total_channels * batches):
            inputs.extend(gen_random_matrix_input(
                *self.total_tup, *self.compute_tup, *self.in_width_tup
            ))
        return inputs

    def assert_all_monitors_empty(self):
        assert self.output_monitor.exp_queue.empty()


    def model(self, inputs):
        # Input reconstruction
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list).reshape(
            -1, self.total_channels, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (2 ** self.IN_FRAC_WIDTH)

        # Float Model
        float_y = self.quantized_model(x)

        # Output beat reconstruction
        y = integer_quantizer_for_hw(float_y, self.OUT_WIDTH, self.OUT_FRAC_WIDTH)
        y = y.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(y.shape[0]):
            model_out.extend(split_matrix(y[i], *self.total_tup, *self.compute_tup))

        return model_out

    def setup_test(self, batches=1):
        inputs = self.generate_inputs(batches=batches)
        self.in_driver.load_driver(inputs)
        exp_out = self.model(inputs)
        self.output_monitor.load_monitor(exp_out)


@cocotb.test()
async def basic(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=2)
    await Timer(100, 'us')
    tb.assert_all_monitors_empty()


@cocotb.test()
async def stream(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=100)
    await Timer(1000, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def backpressure(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    tb.setup_test(batches=200)
    await Timer(1000, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_toggle(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    tb.setup_test(batches=200)
    await Timer(1000, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_backpressure(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    tb.setup_test(batches=200)

    await Timer(1000, 'us')
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    # Consts
    LUT_POW = 5
    ISQRT_WIDTH = 16

    mem_dir = Path(__file__).parent / "build" / "group_norm_2d" / "mem"
    makedirs(mem_dir, exist_ok=True)

    def gen_cfg(
        total_dim0: int = 4,
        total_dim1: int = 4,
        compute_dim0: int = 2,
        compute_dim1: int = 2,
        channels: int = 2,
        in_width: int = 8,
        in_frac_width: int = 4,
        out_width: int = 8,
        out_frac_width: int = 4,
        str_id: str = "default",
    ):
        lut = make_lut(2 ** LUT_POW, ISQRT_WIDTH)
        mem_path = mem_dir / f"lutmem-{str_id}.mem"
        write_memb(mem_path, lut, ISQRT_WIDTH)
        params = {
            "TOTAL_DIM0": total_dim0,
            "TOTAL_DIM1": total_dim1,
            "COMPUTE_DIM0": compute_dim0,
            "COMPUTE_DIM1": compute_dim1,
            "GROUP_CHANNELS": channels,
            "IN_WIDTH": in_width,
            "IN_FRAC_WIDTH": in_frac_width,
            "OUT_WIDTH": out_width,
            "OUT_FRAC_WIDTH": out_frac_width,
            "ISQRT_LUT_MEMFILE": verilator_str_param(str(mem_path)),
        }
        return params

    mase_runner(
        module_param_list=[
            # Default
            gen_cfg(),
            # Rectangle
            gen_cfg(4, 6, 2, 2, 2, 8, 4, 8, 4, "rect0"),
            gen_cfg(6, 2, 2, 2, 2, 8, 4, 8, 4, "rect1"),
            gen_cfg(6, 2, 3, 2, 2, 8, 4, 8, 4, "rect2"),
            gen_cfg(4, 6, 2, 3, 2, 8, 4, 8, 4, "rect3"),
            # Channels
            gen_cfg(4, 4, 2, 2, 1, 8, 4, 8, 4, "channels0"),
            gen_cfg(4, 4, 2, 2, 3, 8, 4, 8, 4, "channels1"),
            # Precision
            gen_cfg(4, 4, 2, 2, 2, 8, 4, 8, 2, "down_frac"),
            gen_cfg(4, 4, 2, 2, 2, 8, 4, 8, 6, "up_frac"),
        ],
        trace=True,
    )
