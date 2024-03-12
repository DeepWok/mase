#!/usr/bin/env python3

import logging
from random import randint
from math import ceil, log2
from copy import copy
# from itertools import batched  # Python 3.12

import torch
import numpy as np
import cocotb
from cocotb.result import TestFailure
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
    random_2d_dimensions,
)

from chop.passes.graph.transforms.quantize.quantized_modules.group_norm2d import (
    _fixed_group_norm_2d_model
)

from mase_components.fixed_arithmetic.test.isqrt_sw import (
    lut_parameter_dict, make_lut
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


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

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready,
            name="Output Monitor",
        )

        # Intermediate value monitors
        self.mu_monitor = StreamMonitor(
            dut.clk, dut.mu_in, dut.mu_acc_valid, dut.mu_acc_ready,
            name="Mu Monitor",
        )
        self.squares_monitor = StreamMonitor(
            dut.clk, dut.square_out, dut.squares_adder_tree_in_valid, dut.squares_adder_tree_in_ready,
            name="Squares Monitor",
        )
        self.sum_squares_monitor = StreamMonitor(
            dut.clk, dut.squares_acc, dut.squares_acc_valid, dut.squares_acc_ready,
            name="Sum Squares Monitor",
        )
        self.var_monitor = StreamMonitor(
            dut.clk, dut.variance_in, dut.squares_acc_valid, dut.squares_acc_ready,
            name="Variance Monitor",
        )
        self.var_clamp_monitor = StreamMonitor(
            dut.clk, dut.variance_clamp_out, dut.variance_clamp_valid,
            dut.variance_clamp_ready, name="Variance Clamp Monitor",
        )
        self.isqrt_monitor = StreamMonitor(
            dut.clk, dut.inv_sqrt_data, dut.inv_sqrt_valid, dut.inv_sqrt_ready,
            name="Inverse Sqrt Monitor",
        )
        self.diff_monitor = StreamMonitor(
            dut.clk, dut.diff_batch_out, dut.diff_batch_out_valid, dut.diff_batch_out_ready,
            name="Diff Monitor",
        )
        self.norm_monitor = StreamMonitor(
            dut.clk, dut.norm_out_data, dut.norm_monitor_valid, dut.norm_monitor_ready,
            name="Norm Monitor",
        )

        # Inverse Square Root LUT
        self.isqrt_lut = make_lut(2**5, 16)

    def generate_inputs(self, num=1):
        inputs = list()
        for _ in range(self.GROUP_CHANNELS * num):
            inputs.extend(gen_random_matrix_input(
                *self.total_tup, *self.compute_tup, *self.in_width_tup
            ))
        return inputs

    def load_intermediate(self, intermediate):
        self.mu_monitor.load_monitor(intermediate["mu"])
        self.var_monitor.load_monitor(intermediate["var"])
        self.var_clamp_monitor.load_monitor(intermediate["var_clamp"])
        self.isqrt_monitor.load_monitor(intermediate["isqrt"])
        self.sum_squares_monitor.load_monitor(intermediate["sum_squares"])
        diff = intermediate["diff"].reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        squares = intermediate["squares"].reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        norm = intermediate["norm"].reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)

        def _load(mon, tensor):
            for i in range(tensor.shape[0]):
                mon.load_monitor(
                    split_matrix(tensor[i], *self.total_tup, *self.compute_tup)
                )
        _load(self.diff_monitor, diff)
        _load(self.squares_monitor, squares)
        _load(self.norm_monitor, norm)

    def assert_all_monitors_empty(self):
        assert self.output_monitor.exp_queue.empty()
        assert self.mu_monitor.exp_queue.empty()
        assert self.var_monitor.exp_queue.empty()
        assert self.var_clamp_monitor.exp_queue.empty()
        assert self.isqrt_monitor.exp_queue.empty()
        assert self.diff_monitor.exp_queue.empty()
        assert self.norm_monitor.exp_queue.empty()
        assert self.squares_monitor.exp_queue.empty()
        assert self.sum_squares_monitor.exp_queue.empty()


    def model(self, inputs):
        # Input reconstruction
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list).reshape(
            -1, self.GROUP_CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (2 ** self.IN_FRAC_WIDTH)

        # Float Model
        norm_out_float, norm_int_out, intermediate = _fixed_group_norm_2d_model(
            x=x,
            in_width=self.IN_WIDTH,
            in_frac_width=self.IN_FRAC_WIDTH,
            diff_width=self.DIFF_WIDTH,
            diff_frac_width=self.DIFF_FRAC_WIDTH,
            square_width=self.SQUARE_WIDTH,
            square_frac_width=self.SQUARE_FRAC_WIDTH,
            variance_width=self.VARIANCE_WIDTH,
            variance_frac_width=self.VARIANCE_FRAC_WIDTH,
            isqrt_width=self.ISQRT_WIDTH,
            isqrt_frac_width=self.ISQRT_FRAC_WIDTH,
            isqrt_lut=self.isqrt_lut,
            norm_width=self.NORM_WIDTH,
            norm_frac_width=self.NORM_FRAC_WIDTH,
            out_width=self.OUT_WIDTH,
            out_frac_width=self.OUT_FRAC_WIDTH,
        )

        # Output beat reconstruction
        y = norm_int_out.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(y.shape[0]):
            model_out.extend(split_matrix(y[i], *self.total_tup, *self.compute_tup))

        return model_out, intermediate

    def setup_test(self, num=1):
        inputs = self.generate_inputs(num=num)
        self.in_driver.load_driver(inputs)
        exp_out, intermediate = self.model(inputs)
        self.output_monitor.load_monitor(exp_out)
        self.load_intermediate(intermediate)


@cocotb.test()
async def basic(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(num=2)
    await Timer(10, 'us')
    tb.assert_all_monitors_empty()


@cocotb.test()
async def stream(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(num=100)
    await Timer(200, 'us')
    assert tb.output_monitor.exp_queue.empty()

# @cocotb.test(skip=True)
# async def long_stream(dut):
#     tb = GroupNorm2dTB(dut)
#     await tb.reset()
#     tb.output_monitor.ready.value = 1

#     inputs = tb.generate_inputs(num=1000)
#     tb.in_driver.load_driver(inputs)
#     exp_out = tb.model(inputs)
#     tb.output_monitor.load_monitor(exp_out)

#     await Timer(2000, 'us')
#     assert tb.output_monitor.exp_queue.empty()


# @cocotb.test(skip=True)
# async def backpressure(dut):
#     tb = GroupNorm2dTB(dut)
#     await tb.reset()
#     cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))

#     inputs = tb.generate_inputs(num=200)
#     tb.in_driver.load_driver(inputs)
#     exp_out = tb.model(inputs)
#     tb.output_monitor.load_monitor(exp_out)

#     await Timer(400, 'us')
#     assert tb.output_monitor.exp_queue.empty()


# @cocotb.test(skip=True)
# async def valid_toggle(dut):
#     tb = GroupNorm2dTB(dut)
#     await tb.reset()
#     tb.output_monitor.ready.value = 1
#     tb.in_driver.set_valid_prob(0.5)

#     inputs = tb.generate_inputs(num=200)
#     tb.in_driver.load_driver(inputs)
#     exp_out = tb.model(inputs)
#     tb.output_monitor.load_monitor(exp_out)

#     await Timer(400, 'us')
#     assert tb.output_monitor.exp_queue.empty()


# @cocotb.test(skip=True)
# async def valid_backpressure(dut):
#     tb = GroupNorm2dTB(dut)
#     await tb.reset()
#     tb.in_driver.set_valid_prob(0.5)
#     cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))

#     inputs = tb.generate_inputs(num=200)
#     tb.in_driver.load_driver(inputs)
#     exp_out = tb.model(inputs)
#     tb.output_monitor.load_monitor(exp_out)

#     await Timer(400, 'us')
#     assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    # def calculate_isqrt_width(
    #     total_dim0, total_dim1, compute_dim0, compute_dim1, group_channels,
    #     in_width, **kwargs
    # ):
    #     depth_dim0 = total_dim0 // compute_dim0
    #     depth_dim1 = total_dim1 // compute_dim1
    #     square_width = in_width * 2
    #     squares_adder_tree_in_size = compute_dim0 * compute_dim1
    #     squares_adder_tree_out_width = ceil(log2(squares_adder_tree_in_size)) + square_width
    #     num_iters = depth_dim0 * depth_dim1 * group_channels
    #     iter_width = ceil(log2(num_iters))
    #     isqrt_width = iter_width + squares_adder_tree_out_width
    #     return isqrt_width


    def gen_cfg():
        compute_dim0, compute_dim1, total_dim0, total_dim1 = random_2d_dimensions()
        params = {
            "TOTAL_DIM0": 4,
            "TOTAL_DIM1": 4,
            "COMPUTE_DIM0": 2,
            "COMPUTE_DIM1": 2,
            "GROUP_CHANNELS": 2, # randint(1, 4),
            "IN_WIDTH": 8,
            "IN_FRAC_WIDTH": 4,
            "OUT_WIDTH": 8,
            "OUT_FRAC_WIDTH": 4,
        }
        # lower_case_keys = {k.lower(): v for k, v in params.items()}
        # lut_width = calculate_isqrt_width(**lower_case_keys)
        params |= lut_parameter_dict(2**5, 16)
        return params

    mase_runner(
        module_param_list=[gen_cfg()],
        trace=True,
    )
