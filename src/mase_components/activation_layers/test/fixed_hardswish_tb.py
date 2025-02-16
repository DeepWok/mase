#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
import pdb
import cocotb
from functools import partial
from cocotb.triggers import *
from chop.nn.quantizers import integer_quantizer
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

import pytest


class Hardswishtb(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "DATA_IN_0_PRECISION_0",
                "DATA_IN_0_PRECISION_1",
                "DATA_IN_0_TENSOR_SIZE_DIM_0",
                "DATA_IN_0_TENSOR_SIZE_DIM_1",
                "DATA_IN_0_PARALLELISM_DIM_0",
                "DATA_IN_0_PARALLELISM_DIM_1",
                "DATA_OUT_0_PRECISION_0",
                "DATA_OUT_0_PRECISION_1",
                "DATA_OUT_0_TENSOR_SIZE_DIM_0",
                "DATA_OUT_0_TENSOR_SIZE_DIM_1",
                "DATA_OUT_0_PARALLELISM_DIM_0",
                "DATA_OUT_0_PARALLELISM_DIM_1",
            ]
        )
        self.width = 8
        self.fracw = 4
        self.samples = 10

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )
        self.thresh = 0.5

    def exp(self, inputs):
        # Run the model with the provided inputs and return the outputs
        tmp0 = 3 * 2**self.fracw
        tmp1 = inputs + tmp0
        tmp2 = tmp1 * (2**-3) + tmp1 * (2**-4)
        # qtmps = self.dquantizer(tmp2)
        tmp3 = tmp2 * inputs
        unsignedout = torch.where(tmp3 < 0, torch.tensor(tmp3 % (2**self.width)), tmp3)
        # return unsignedout.tolist()
        return unsignedout

    def generate_inputs(self, w, fracw):
        self.dquantizer = partial(
            integer_quantizer, width=self.width, frac_width=self.fracw
        )
        realinp = torch.randn(self.samples)
        inputs = self.dquantizer(realinp)
        intinp = (inputs * 2**self.fracw).to(torch.int64)
        intinp.clamp(
            min=-(2 ** (self.width - self.fracw - 1)),
            max=2 ** (self.width - self.fracw - 1) - 1,
        )
        return intinp


@cocotb.test()
async def cocotb_test(dut):
    tb = Hardswishtb(dut)
    await tb.reset()
    logger.info(f"Reset finished")
    tb.data_out_0_monitor.ready.value = 1

    inputs = tb.generate_inputs(tb.width, tb.fracw)
    # logger.info(f"inputs: {inputs}, q_inputs: {q_inputs}")
    exp_out = tb.exp(inputs)

    tb.data_in_0_driver.append(inputs.tolist())
    # To do: replace with tb.load_monitors(exp_out)
    tb.data_out_0_monitor.expect(exp_out)
    # To do: replace with tb.run()
    await Timer(10, units="us")
    # To do: replace with tb.monitors_done() --> for monitor, call monitor_done()
    assert tb.data_out_0_monitor.exp_queue.empty()


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_hardswish():
    mase_runner(
        module_param_list=[
            {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 10,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 4,
                "DATA_OUT_0_PRECISION_0": 8,
                "DATA_OUT_0_PRECISION_1": 4,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 10,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
            }
        ]
    )


if __name__ == "__main__":
    test_fixed_hardswish()
