#!/usr/bin/env python3

import logging
import torch
from torch import Tensor

import random
from random import randint
from math import ceil, log2
from copy import copy

import numpy as np

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_tools import gen_random_matrix_input, matrix_mult_model

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class ChainMatmulTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "N",
                "M",
                "K",
                "Z",
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "INT_WIDTH",
                "INT_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                "COMPUTE_DIM0",
                "COMPUTE_DIM1",
                "SYMMETRIC",
            ]
        )

        # Drivers & Monitors
        self.a_driver = StreamDriver(dut.clk, dut.a_data, dut.a_valid, dut.a_ready)
        self.b_driver = StreamDriver(dut.clk, dut.b_data, dut.b_valid, dut.b_ready)
        self.c_driver = StreamDriver(dut.clk, dut.c_data, dut.c_valid, dut.c_ready)
        self.d_monitor = StreamMonitor(
            dut.clk, dut.d_data, dut.d_valid, dut.d_ready, check=True
        )

    def cleanup(self):
        self.a_driver.kill()
        self.b_driver.kill()
        self.c_driver.kill()
        self.d_monitor.kill()

    # Dimensions for chain matmul are: (nm * mk) * kz = nz
    def generate_inputs(self):
        A_inputs = gen_random_matrix_input(
            self.M,
            self.N,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
        )
        B_inputs = gen_random_matrix_input(
            self.K,
            self.M,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
        )
        C_inputs = gen_random_matrix_input(
            self.Z,
            self.K,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
        )
        return A_inputs, B_inputs, C_inputs

    def model(self, A_inputs, B_inputs, C_inputs):
        # (nm * mk) -> nk
        intermediate_matrix = matrix_mult_model(
            self.M,
            self.N,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.K,
            self.M,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.K,
            self.N,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
            self.INT_WIDTH,
            self.INT_FRAC_WIDTH,
            self.SYMMETRIC,
            A_inputs,
            B_inputs,
        )
        # (nk * kz) -> nz
        output = matrix_mult_model(
            self.K,
            self.N,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.Z,
            self.K,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.Z,
            self.N,
            self.COMPUTE_DIM0,
            self.COMPUTE_DIM1,
            self.INT_WIDTH,
            self.INT_FRAC_WIDTH,
            self.IN_WIDTH,
            self.IN_FRAC_WIDTH,
            self.OUT_WIDTH,
            self.OUT_FRAC_WIDTH,
            self.SYMMETRIC,
            intermediate_matrix,
            C_inputs,
        )
        return output


@cocotb.test()
async def cocotb_test_basic(dut):
    tb = ChainMatmulTB(dut)
    tb.d_monitor.ready.value = 1
    await tb.reset()
    A_inputs, B_inputs, C_inputs = tb.generate_inputs()
    exp_out = tb.model(A_inputs, B_inputs, C_inputs)

    # Setup drivers and monitors
    for a in A_inputs:
        tb.a_driver.append(a)
    for b in B_inputs:
        tb.b_driver.append(b)
    for c in C_inputs:
        tb.c_driver.append(c)
    for o in exp_out:
        tb.d_monitor.expect(o)
    await Timer(100, units="us")
    assert tb.d_monitor.exp_queue.empty()
    tb.cleanup()


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_chain_matmul():
    mase_runner(
        module_param_list=[
            {"N": 2, "M": 2, "K": 2, "Z": 2},
            {"N": 4, "M": 2, "K": 4, "Z": 2},
            {"N": 2, "M": 4, "K": 2, "Z": 8},
            {"N": 8, "M": 2, "K": 8, "Z": 2},
            {"N": 8, "M": 4, "K": 4, "Z": 2},
        ]
    )


if __name__ == "__main__":
    test_chain_matmul()
