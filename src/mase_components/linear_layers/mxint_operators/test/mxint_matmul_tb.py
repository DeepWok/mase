#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging, traceback, pdb, sys

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)

from mase_cocotb.runner import mase_runner
from utils import block_mxint_quant
from mase_cocotb.matrix_tools import gen_random_matrix_input, matrix_mult_model
from mase_cocotb.utils import bit_driver

from random import randint
import torch
from math import ceil, log2
import random

import pytest

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


sys.excepthook = excepthook
torch.manual_seed(10)


class MXIntMatmulTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = 1
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.log.setLevel(logging.DEBUG)
        self.a_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.ma_data, dut.ea_data),
            dut.a_valid,
            dut.a_ready,
        )
        self.b_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mb_data, dut.eb_data),
            dut.b_valid,
            dut.b_ready,
        )

        self.output_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mout_data, dut.eout_data),
            dut.out_valid,
            dut.out_ready,
            check=True,
        )

    def generate_inputs(self):
        for _ in range(self.num):
            a_random = 20 * torch.rand(
                self.get_parameter("A_TOTAL_DIM1"), self.get_parameter("A_TOTAL_DIM0")
            )
            (qa, ma, ea) = block_mxint_quant(
                a_random,
                q_config={
                    "width": self.get_parameter("A_MAN_WIDTH"),
                    "exponent_width": self.get_parameter("A_EXP_WIDTH"),
                },
                parallelism=[
                    self.get_parameter("A_COMPUTE_DIM0"),
                    self.get_parameter("A_COMPUTE_DIM1"),
                ],
            )
            b_random = 20 * torch.rand(
                self.get_parameter("B_TOTAL_DIM1"), self.get_parameter("B_TOTAL_DIM0")
            )
            (qb, mb, eb) = block_mxint_quant(
                b_random,
                q_config={
                    "width": self.get_parameter("B_MAN_WIDTH"),
                    "exponent_width": self.get_parameter("B_EXP_WIDTH"),
                },
                parallelism=[
                    self.get_parameter("B_COMPUTE_DIM1"),
                    self.get_parameter("B_COMPUTE_DIM0"),
                ],
            )
            matmul_out = qa @ qb
            self.log.debug(f"hardware_out = {ma @ mb}")
            (qout, mout, eout) = block_mxint_quant(
                matmul_out,
                q_config={
                    "width": self.get_parameter("OUT_MAN_WIDTH"),
                    "exponent_width": self.get_parameter("OUT_EXP_WIDTH"),
                },
                parallelism=[
                    self.get_parameter("C_COMPUTE_DIM1"),
                    self.get_parameter("C_COMPUTE_DIM0"),
                ],
            )
            from utils import pack_tensor_to_mx_listed_chunk

            a_inputs = pack_tensor_to_mx_listed_chunk(
                ma,
                ea,
                parallelism=[
                    self.get_parameter("A_COMPUTE_DIM0"),
                    self.get_parameter("A_COMPUTE_DIM1"),
                ],
            )
            b_inputs = pack_tensor_to_mx_listed_chunk(
                mb,
                eb,
                parallelism=[
                    self.get_parameter("B_COMPUTE_DIM1"),
                    self.get_parameter("B_COMPUTE_DIM0"),
                ],
            )
            exp_outputs = pack_tensor_to_mx_listed_chunk(
                mout,
                eout,
                parallelism=[
                    self.get_parameter("C_COMPUTE_DIM1"),
                    self.get_parameter("C_COMPUTE_DIM0"),
                ],
            )
        return a_inputs, b_inputs, exp_outputs

    async def run_test(self, batches, us):
        await self.reset()
        self.num = batches
        logger.info(f"Reset finished")
        self.output_monitor.ready.value = 1

        logger.info(f"generating inputs")
        a_inputs, b_inputs, exp_outputs = self.generate_inputs()
        print(a_inputs)
        print(b_inputs)
        # Load the inputs driver
        self.a_driver.load_driver(a_inputs)
        self.b_driver.load_driver(b_inputs)
        # Load the output monitor
        self.output_monitor.load_monitor(exp_outputs)

        await Timer(us, units="us")
        assert self.output_monitor.exp_queue.empty()


# @cocotb.test()
# async def single_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1000, us=2000)


# @cocotb.test()
# async def repeated_mult_backpressure(dut):
#     tb = MXIntMatmulTB(dut)
#     cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))
#     await tb.run_test(batches=500, us=2000)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = MXIntMatmulTB(dut)
    tb.a_driver.set_valid_prob(0.7)
    tb.b_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))
    await tb.run_test(batches=20, us=200)


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
        "A_MAN_WIDTH": 8,
        "A_EXP_WIDTH": 3,
        "B_MAN_WIDTH": 8,
        "B_EXP_WIDTH": 3,
        "OUT_MAN_WIDTH": 8,
        "OUT_EXP_WIDTH": 3,
    }

    mase_runner(
        module_param_list=[
            # Default Square
            DEFAULT_CONFIG,
            #
            # {
            #     **DEFAULT_CONFIG,
            #     "A_MAN_WIDTH": 9,
            #     "A_EXP_WIDTH": 3,
            #     "B_MAN_WIDTH": 16,
            #     "B_EXP_WIDTH": 4,
            #     "OUT_MAN_WIDTH": 12,
            #     "OUT_EXP_WIDTH": 4,
            # },
            # # Long Rectangle, should saturate many values
            {
                **DEFAULT_CONFIG,
                "A_TOTAL_DIM0": 16,
                "A_TOTAL_DIM1": 4,
                "B_TOTAL_DIM0": 4,
                "B_TOTAL_DIM1": 16,
                "A_COMPUTE_DIM0": 4,
                "A_COMPUTE_DIM1": 4,
                "B_COMPUTE_DIM0": 4,
                "B_COMPUTE_DIM1": 4,  # Must equal A_COMPUTE_DIM0
            },
            # # # Change window to full size
            {
                **DEFAULT_CONFIG,
                "A_COMPUTE_DIM0": 4,
                "A_COMPUTE_DIM1": 4,
                "B_COMPUTE_DIM0": 4,
                "B_COMPUTE_DIM1": 4,
            },
            # # Dimensions
            # *generate_random_dimension_cfg([DEFAULT_CONFIG]),
        ],
        trace=True,
        jobs=12,
        extra_build_args=["--trace-depth", "5"],
    )


if __name__ == "__main__":
    test_matmul()
