#!/usr/bin/env python3

import logging, pytest
from random import randint

import numpy as np
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

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class TransposeTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)
        self.assign_self_params(["WIDTH", "DIM0", "DIM1"])

    def generate_inputs(self):
        # Compute dimensions = total dimensions
        inputs = gen_random_matrix_input(
            self.DIM0, self.DIM1, self.DIM0, self.DIM1, self.WIDTH, 0  # 0 frac width
        )
        return inputs[0]

    def model(self, inputs):
        A = rebuild_matrix([inputs], self.DIM0, self.DIM1, self.DIM0, self.DIM1)
        # Need to reverse the dimensions due to transpose
        # DIM0 becomes DIM1 in the new matrix and vice versa
        AT = split_matrix(A.T, self.DIM1, self.DIM0, self.DIM1, self.DIM0)[0]
        return AT


@cocotb.test()
async def cocotb_test_transpose(dut):
    tb = TransposeTB(dut)
    for _ in range(1000):
        in_data = tb.generate_inputs()
        dut.in_data.value = in_data
        await Timer(2, "ns")
        exp_out = tb.model(in_data)
        out_data = [int(x) for x in dut.out_data.value]
        assert np.array_equal(exp_out, out_data)


def generate_random_params(num=3):
    cfgs = list()
    for _ in range(num):
        cfgs.append(
            {
                "WIDTH": randint(1, 16),
                "DIM0": randint(2, 12),
                "DIM1": randint(2, 12),
            }
        )
    return cfgs


@pytest.mark.dev
def test_transpose():
    mase_runner(module_param_list=generate_random_params())


if __name__ == "__main__":
    test_transpose()
