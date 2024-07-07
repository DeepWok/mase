#!/usr/bin/env python3

import logging, pytest
from random import randint

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_tools import gen_random_matrix_input, matrix_mult_model
from mase_cocotb.utils import bit_driver


logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class MatmulTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "A_TOTAL_DIM0",
                "A_TOTAL_DIM1",
                "B_TOTAL_DIM0",
                "B_TOTAL_DIM1",
                "C_TOTAL_DIM0",
                "C_TOTAL_DIM1",
                "A_COMPUTE_DIM0",
                "A_COMPUTE_DIM1",
                "B_COMPUTE_DIM0",
                "B_COMPUTE_DIM1",
                "C_COMPUTE_DIM0",
                "C_COMPUTE_DIM1",
                "A_WIDTH",
                "A_FRAC_WIDTH",
                "B_WIDTH",
                "B_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                "OUT_SYMMETRIC",
            ]
        )

        # Drivers & Monitors
        self.a_driver = StreamDriver(dut.clk, dut.a_data, dut.a_valid, dut.a_ready)
        self.b_driver = StreamDriver(dut.clk, dut.b_data, dut.b_valid, dut.b_ready)
        self.output_monitor = StreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            check=True,
            unsigned=True,
        )

    def generate_inputs(self):
        A_inputs = gen_random_matrix_input(
            self.A_TOTAL_DIM0,
            self.A_TOTAL_DIM1,
            self.A_COMPUTE_DIM0,
            self.A_COMPUTE_DIM1,
            self.A_WIDTH,
            self.A_FRAC_WIDTH,
        )
        B_inputs = gen_random_matrix_input(
            self.B_TOTAL_DIM0,
            self.B_TOTAL_DIM1,
            self.B_COMPUTE_DIM0,
            self.B_COMPUTE_DIM1,
            self.B_WIDTH,
            self.B_FRAC_WIDTH,
        )
        return A_inputs, B_inputs

    def model(self, A_inputs, B_inputs):
        return matrix_mult_model(
            self.A_TOTAL_DIM0,
            self.A_TOTAL_DIM1,
            self.A_COMPUTE_DIM0,
            self.A_COMPUTE_DIM1,
            self.B_TOTAL_DIM0,
            self.B_TOTAL_DIM1,
            self.B_COMPUTE_DIM0,
            self.B_COMPUTE_DIM1,
            self.C_TOTAL_DIM0,
            self.C_TOTAL_DIM1,
            self.C_COMPUTE_DIM0,
            self.C_COMPUTE_DIM1,
            self.A_WIDTH,
            self.A_FRAC_WIDTH,
            self.B_WIDTH,
            self.B_FRAC_WIDTH,
            self.OUT_WIDTH,
            self.OUT_FRAC_WIDTH,
            self.OUT_SYMMETRIC,
            A_inputs,
            B_inputs,
        )

    async def run_test(self, batches, us):
        await self.reset()
        for _ in range(batches):
            A_inputs, B_inputs = self.generate_inputs()
            exp_out = self.model(A_inputs, B_inputs)
            # Setup drivers and monitors
            self.a_driver.load_driver(A_inputs)
            self.b_driver.load_driver(B_inputs)
            self.output_monitor.load_monitor(exp_out)
        await Timer(us, units="us")
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def single_mult(dut):
    tb = MatmulTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=1, us=100)


@cocotb.test()
async def repeated_mult(dut):
    tb = MatmulTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def repeated_mult_backpressure(dut):
    tb = MatmulTB(dut)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))
    await tb.run_test(batches=500, us=2000)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = MatmulTB(dut)
    tb.a_driver.set_valid_prob(0.7)
    tb.b_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))
    await tb.run_test(batches=500, us=2000)


def gen_random_dimensions():
    compute_dim0 = randint(2, 3)
    compute_dim1 = randint(2, 3)
    total_dim0 = compute_dim0 * randint(1, 3)
    total_dim1 = compute_dim1 * randint(1, 3)
    return compute_dim0, compute_dim1, total_dim0, total_dim1


def random_matrix_mult_dim_cfg():
    a_cfg = gen_random_dimensions()
    a_compute_dim0, a_compute_dim1, a_total_dim0, a_total_dim1 = a_cfg
    b_cfg = gen_random_dimensions()
    b_compute_dim0, _, b_total_dim0, _ = b_cfg
    return {
        "A_TOTAL_DIM0": a_total_dim0,
        "A_TOTAL_DIM1": a_total_dim1,
        "B_TOTAL_DIM0": b_total_dim0,
        "B_TOTAL_DIM1": a_total_dim0,  # Must equal A_TOTAL_DIM0
        "A_COMPUTE_DIM0": a_compute_dim0,
        "A_COMPUTE_DIM1": a_compute_dim1,
        "B_COMPUTE_DIM0": b_compute_dim0,
        "B_COMPUTE_DIM1": a_compute_dim0,  # Must equal A_COMPUTE_DIM0
    }


def generate_random_dimension_cfg(cfg_list, multiple=3):
    new_cfgs = list()
    for cfg in cfg_list:
        new_cfgs.extend(
            [{**cfg, **random_matrix_mult_dim_cfg()} for _ in range(multiple)]
        )
    return new_cfgs


@pytest.mark.dev
def test_matmul():
    # Default is a square matrix mult
    # 4x4 4x4 matrix multiplication done using 2x2 window
    DEFAULT_CONFIG = {
        "A_TOTAL_DIM0": 4,
        "A_TOTAL_DIM1": 4,
        "B_TOTAL_DIM0": 4,
        "B_TOTAL_DIM1": 4,  # Must equal A_TOTAL_DIM0
        "A_COMPUTE_DIM0": 2,
        "A_COMPUTE_DIM1": 2,
        "B_COMPUTE_DIM0": 2,
        "B_COMPUTE_DIM1": 2,  # Must equal A_COMPUTE_DIM0
        "A_WIDTH": 8,
        "A_FRAC_WIDTH": 1,
        "B_WIDTH": 8,
        "B_FRAC_WIDTH": 1,
        "OUT_WIDTH": 16,
        "OUT_FRAC_WIDTH": 2,
    }

    mase_runner(
        module_param_list=[
            # Default Square
            DEFAULT_CONFIG,
            # Failing case before
            {
                **DEFAULT_CONFIG,
                "A_TOTAL_DIM0": 4,
                "A_TOTAL_DIM1": 4,
                "B_TOTAL_DIM0": 2,
                "B_TOTAL_DIM1": 4,
                "A_WIDTH": 4,
                "A_FRAC_WIDTH": 1,
                "B_WIDTH": 4,
                "B_FRAC_WIDTH": 1,
                "OUT_WIDTH": 8,
                "OUT_FRAC_WIDTH": 1,
            },
            # Long Rectangle, should saturate many values
            {
                **DEFAULT_CONFIG,
                "A_TOTAL_DIM0": 16,
                "A_TOTAL_DIM1": 2,
                "B_TOTAL_DIM0": 2,
                "B_TOTAL_DIM1": 16,
                "OUT_WIDTH": 10,
                "OUT_FRAC_WIDTH": 0,
            },
            # Change window to full size
            {
                **DEFAULT_CONFIG,
                "A_COMPUTE_DIM0": 4,
                "A_COMPUTE_DIM1": 4,
                "B_COMPUTE_DIM0": 4,
                "B_COMPUTE_DIM1": 4,
            },
            # Dimensions
            *generate_random_dimension_cfg([DEFAULT_CONFIG]),
        ],
        trace=True,
        jobs=12,
    )


if __name__ == "__main__":
    test_matmul()
