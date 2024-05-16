#!/usr/bin/env python3

import logging
from random import randint

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MatrixStreamTransposeTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            ["TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1", "DATA_WIDTH"]
        )

        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.out_monitor = StreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            check=True,
            unsigned=True,
        )

    def generate_inputs(self):
        return gen_random_matrix_input(
            self.TOTAL_DIM0,
            self.TOTAL_DIM1,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.DATA_WIDTH,
            0,
        )

    def model(self, X):
        # Note: Dimensions are transposed
        X_matrix = rebuild_matrix(
            X, self.TOTAL_DIM0, self.TOTAL_DIM1, self.COMPUTE_DIM0, self.COMPUTE_DIM1
        )
        return split_matrix(
            X_matrix.T,
            self.TOTAL_DIM1,
            self.TOTAL_DIM0,
            self.COMPUTE_DIM1,
            self.COMPUTE_DIM0,
        )


@cocotb.test()
async def single_transpose(dut):
    tb = MatrixStreamTransposeTB(dut)
    await tb.reset()
    tb.out_monitor.ready.value = 1
    X = tb.generate_inputs()
    Y = tb.model(X)
    for x in X:
        tb.in_driver.append(x)
    for y in Y:
        tb.out_monitor.expect(y)
    await Timer(100, units="us")
    assert tb.out_monitor.exp_queue.empty()


@cocotb.test()
async def multiple_transpose(dut):
    tb = MatrixStreamTransposeTB(dut)
    await tb.reset()
    tb.out_monitor.ready.value = 1
    for _ in range(100):
        X = tb.generate_inputs()
        Y = tb.model(X)
        for x in X:
            tb.in_driver.append(x)
        for y in Y:
            tb.out_monitor.expect(y)
    await Timer(1000, units="us")
    assert tb.out_monitor.exp_queue.empty()


@cocotb.test()
async def multiple_transpose_backpressure(dut):
    tb = MatrixStreamTransposeTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))
    for _ in range(100):
        X = tb.generate_inputs()
        Y = tb.model(X)
        for x in X:
            tb.in_driver.append(x)
        for y in Y:
            tb.out_monitor.expect(y)
    await Timer(1000, units="us")
    assert tb.out_monitor.exp_queue.empty()


@cocotb.test()
async def multiple_transpose_valid_backpressure(dut):
    tb = MatrixStreamTransposeTB(dut)
    tb.in_driver.set_valid_prob(0.6)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))
    for _ in range(100):
        X = tb.generate_inputs()
        Y = tb.model(X)
        for x in X:
            tb.in_driver.append(x)
        for y in Y:
            tb.out_monitor.expect(y)
    await Timer(2000, units="us")
    assert tb.out_monitor.exp_queue.empty()


def gen_random_params():
    compute_dim0 = randint(2, 3)
    compute_dim1 = randint(2, 3)
    return {
        "TOTAL_DIM0": compute_dim0 * randint(1, 3),
        "TOTAL_DIM1": compute_dim1 * randint(1, 3),
        "COMPUTE_DIM0": compute_dim0,
        "COMPUTE_DIM1": compute_dim1,
        "DATA_WIDTH": randint(2, 10),
    }


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_matrix_stream_transpose():
    # Run tests with different params
    mase_runner(
        module_param_list=[
            # Square Test
            {
                "TOTAL_DIM0": 4,
                "TOTAL_DIM1": 4,
                "COMPUTE_DIM0": 2,
                "COMPUTE_DIM1": 2,
                "DATA_WIDTH": randint(2, 10),
            },
            # Rectangle Test
            {
                "TOTAL_DIM0": 4,
                "TOTAL_DIM1": 2,
                "COMPUTE_DIM0": 2,
                "COMPUTE_DIM1": 2,
                "DATA_WIDTH": randint(2, 10),
            },
            # Random test
            *[gen_random_params() for _ in range(5)],
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_matrix_stream_transpose()
