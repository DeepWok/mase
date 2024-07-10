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
from torch.autograd.function import InplaceFunction


import torch
import pytest

from chop.nn.quantizers import integer_quantizer

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def get_in_and_out(x, fn, width, frac_width):
    ins = integer_quantizer(x, width=width, frac_width=frac_width)
    y = fn(x)
    outs = integer_quantizer(y, width=width, frac_width=frac_width)
    outs = outs * 2**frac_width
    outs = outs.int()
    ins = ins * 2**frac_width
    ins = ins.int()
    return (ins, outs)


class LeakyReLUTB(Testbench):
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
        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

    def generate_inputs_outputs(self, width, frac_width, negative_slope):
        inputs = torch.tensor([10, 10, -4, -4]).float()
        fn = torch.nn.LeakyReLU(negative_slope=negative_slope)
        ins, outs = get_in_and_out(inputs, fn, width, frac_width)
        print(ins, outs)
        return ins, outs


@cocotb.test()
async def cocotb_test(dut):
    # pdb.set_trace()
    tb = LeakyReLUTB(dut)
    await tb.reset()
    logger.info(f"Reset finished")
    tb.data_out_0_monitor.ready.value = 1

    inputs, exp_outs = tb.generate_inputs_outputs(8, 4, 2**-4)

    tb.data_in_0_driver.append(inputs.tolist())

    tb.data_out_0_monitor.expect(exp_outs.tolist())
    await Timer(1000, units="us")
    assert tb.data_out_0_monitor.exp_queue.empty()


@pytest.mark.skip(reason="Needs to be fixed.")
def test_fixed_leaky_relu():
    # The behavior of leaky relu may not exactly match the behavior of the torch leaky relu.
    # This is becase of the multiplication with the negative slope is floored in system verilog instead of rounded.
    mase_runner(
        module_param_list=[
            {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 4,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_IN_0_PRECISION_0": 16,
                "DATA_IN_0_PRECISION_1": 4,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 4,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_PRECISION_0": 16,
                "DATA_OUT_0_PRECISION_1": 4,
                "NEGATIVE_SLOPE_PRECISION_0": 16,
                "NEGATIVE_SLOPE_PRECISION_1": 4,
                "NEGATIVE_SLOPE_VALUE": 1,
            }
        ]
    )


if __name__ == "__main__":
    test_fixed_leaky_relu()
