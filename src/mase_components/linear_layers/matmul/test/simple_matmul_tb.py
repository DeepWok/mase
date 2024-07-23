#!/usr/bin/env python3

import logging
import torch
import pytest

from random import randint
import math
from copy import copy

import numpy as np

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class SimpleMatMulTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "N",
                "M",
                "K",
                "X_WIDTH",
                "Y_WIDTH",
                "OUT_WIDTH",
                "X_FRAC_WIDTH",
                "Y_FRAC_WIDTH",
                "OUT_FRAC_WIDTH",
            ]
        )

        self.X_MAX = 2**self.X_WIDTH - 1
        self.Y_MAX = 2**self.Y_WIDTH - 1

        self.x_driver = StreamDriver(dut.clk, dut.x_data, dut.x_valid, dut.x_ready)
        self.y_driver = StreamDriver(dut.clk, dut.y_data, dut.y_valid, dut.y_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready, unsigned=True
        )

    def generate_inputs(self, random=False):
        if random:
            # Generate random numbers between [-1, 1]
            X = (torch.rand(size=(self.N, self.M)) - 0.5) * 2
            Y = (torch.rand(size=(self.M, self.K)) - 0.5) * 2

            # Scale up to entire range of numbers
            X *= 2 ** (self.X_WIDTH - self.X_FRAC_WIDTH - 1)
            Y *= 2 ** (self.Y_WIDTH - self.Y_FRAC_WIDTH - 1)
        else:
            X = torch.arange(self.N * self.M).reshape(self.N, self.M)
            Y = torch.arange(self.M * self.K).reshape(self.M, self.K)

        X = quantize_to_int(X, self.X_WIDTH, self.X_FRAC_WIDTH)
        Y = quantize_to_int(Y, self.Y_WIDTH, self.Y_FRAC_WIDTH)

        X_input = X.flatten().tolist()
        X_input.reverse()
        Y_input = Y.flatten().tolist()
        Y_input.reverse()
        return X_input, Y_input

    def model(self, X, Y):
        X_input = copy(X)
        Y_input = copy(Y)

        X_input.reverse()
        Y_input.reverse()

        X_input = torch.tensor(X_input).reshape(self.N, self.M)
        Y_input = torch.tensor(Y_input).reshape(self.M, self.K)

        logger.debug("Original")
        logger.debug(X_input)
        logger.debug(Y_input)

        logger.debug("Sign Extended & Scaled")
        X_input = sign_extend_t(X_input, self.X_WIDTH).type(torch.float32) / (
            2**self.X_FRAC_WIDTH
        )
        Y_input = sign_extend_t(Y_input, self.Y_WIDTH).type(torch.float32) / (
            2**self.Y_FRAC_WIDTH
        )
        logger.debug(X_input)
        logger.debug(Y_input)

        output = torch.matmul(X_input, Y_input)
        output = quantize_to_int(output, self.OUT_WIDTH, self.OUT_FRAC_WIDTH)

        logger.debug("Output")
        logger.debug(output)

        output = output.flatten().tolist()
        output.reverse()
        return output


@cocotb.test()
async def small_positive_nums(dut):
    """Basic multiplication with small numbers"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    X, Y = tb.generate_inputs()
    exp_out = tb.model(X, Y)
    tb.x_driver.append(X)
    tb.y_driver.append(Y)
    tb.output_monitor.expect(exp_out)
    await Timer(1, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def repeated_multiply(dut):
    """Repeated multiplication with small positive numbers"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    for _ in range(100):
        X, Y = tb.generate_inputs()
        exp_out = tb.model(X, Y)
        tb.x_driver.append(X)
        tb.y_driver.append(Y)
        tb.output_monitor.expect(exp_out)
    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_multiply(dut):
    """Single multiplication with random floats"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    X, Y = tb.generate_inputs(random=True)
    exp_out = tb.model(X, Y)
    tb.x_driver.append(X)
    tb.y_driver.append(Y)
    tb.output_monitor.expect(exp_out)
    await Timer(1, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_repeated_multiply(dut):
    """Many multiplications with random floats"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    for _ in range(100):
        X, Y = tb.generate_inputs(random=True)
        exp_out = tb.model(X, Y)
        tb.x_driver.append(X)
        tb.y_driver.append(Y)
        tb.output_monitor.expect(exp_out)
    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_repeated_multiply_backpressure(dut):
    """Many multiplications with random floats and backpressure"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()

    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))

    for _ in range(100):
        X, Y = tb.generate_inputs(random=True)
        exp_out = tb.model(X, Y)
        tb.x_driver.append(X)
        tb.y_driver.append(Y)
        tb.output_monitor.expect(exp_out)
    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_repeated_multiply_valid(dut):
    """Many multiplications with random floats"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.x_driver.set_valid_prob(0.7)
    tb.y_driver.set_valid_prob(0.7)
    tb.output_monitor.ready.value = 1

    for _ in range(100):
        X, Y = tb.generate_inputs(random=True)
        exp_out = tb.model(X, Y)
        tb.x_driver.append(X)
        tb.y_driver.append(Y)
        tb.output_monitor.expect(exp_out)
    await Timer(200, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_repeated_multiply_valid_backpressure(dut):
    """Many multiplications with random floats"""
    tb = SimpleMatMulTB(dut)
    await tb.reset()
    tb.x_driver.set_valid_prob(0.7)
    tb.y_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))

    for _ in range(100):
        X, Y = tb.generate_inputs(random=True)
        exp_out = tb.model(X, Y)
        tb.x_driver.append(X)
        tb.y_driver.append(Y)
        tb.output_monitor.expect(exp_out)
    await Timer(300, units="us")
    assert tb.output_monitor.exp_queue.empty()


def generate_random_dimensions(low, high):
    return {
        "N": randint(low, high),
        "M": randint(low, high),
        "K": randint(low, high),
    }


def generate_random_widths():
    widths = {
        "X_WIDTH": randint(4, 8),
        "Y_WIDTH": randint(4, 8),
        "OUT_WIDTH": randint(4, 8),
    }
    frac_widths = {
        "X_FRAC_WIDTH": randint(0, widths["X_WIDTH"]),
        "Y_FRAC_WIDTH": randint(0, widths["Y_WIDTH"]),
        "OUT_FRAC_WIDTH": randint(0, widths["OUT_WIDTH"]),
    }
    return {**widths, **frac_widths}


@pytest.mark.dev
def test_simple_matmul():
    # Run tests with different params
    mase_runner(
        module_param_list=[
            {"N": 2, "M": 2, "K": 2},
            {"N": 2, "M": 3, "K": 4},
            {"N": 1, "M": 10, "K": 1},
            *[
                {**generate_random_dimensions(2, 4), **generate_random_widths()}
                for _ in range(5)
            ],
        ]
    )


if __name__ == "__main__":
    test_simple_matmul()
