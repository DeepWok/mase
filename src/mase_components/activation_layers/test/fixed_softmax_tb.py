#!/usr/bin/env python3

import os
import pytest

import torch
import logging
from functools import partial
from src.mase_components.helper import generate_memory
from pathlib import Path
import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import fixed_preprocess_tensor

from mase_cocotb.utils import bit_driver
from chop.nn.quantized.functional import softmax_integer

class SoftmaxTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.in_data_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        self.out_data_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        # Model
        self.model = partial(
            softmax_integer,
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "data_in_exp_width": self.get_parameter("DATA_EXP_0_PRECISION_0"),
                "data_in_exp_frac_width": self.get_parameter("DATA_EXP_0_PRECISION_1"),
                "data_out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                "mult_data": CONSTANT_MULT,
            },
            dim=-1,
            floor=True,
        )

        # Set verbosity of driver and monitor loggers to debug
        self.in_data_driver.log.setLevel(logging.DEBUG)
        self.out_data_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        return torch.randn(
            (
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, batches, us):
        await self.reset()
        self.log.info(f"Reset finished")

        for _ in range(batches):
            inputs = self.generate_inputs()
            exp_out = self.model(inputs)

            # * Load the inputs driver
            self.log.info(f"Processing inputs: {inputs}")
            inputs = fixed_preprocess_tensor(
                tensor=inputs,
                q_config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                    self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
                ],
                floor=True,
            )
            self.in_data_driver.load_driver(inputs)

            # * Load the output monitor
            self.log.info(f"Processing outputs: {exp_out}")
            outs = fixed_preprocess_tensor(
                tensor=exp_out,
                q_config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                    self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
                ],
            )
            self.out_data_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.out_data_monitor.exp_queue.empty()

@cocotb.test()
async def single_test(dut):
    tb = SoftmaxTB(dut)
    tb.out_data_monitor.ready.value = 1
    await tb.run_test(batches=50, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = SoftmaxTB(dut)
#     tb.out_data_monitor.ready.value = 1
#     await tb.run_test(batches=100, us=2000)


# @cocotb.test()
# async def repeated_mult_backpressure(dut):
#     tb = SoftmaxTB(dut)
#     cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
#     await tb.run_test(batches=10, us=500)


# @cocotb.test()
# async def repeated_mult_valid_backpressure(dut):
#     tb = SoftmaxTB(dut)
#     tb.in_data_driver.set_valid_prob(0.7)
#     cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
#     await tb.run_test(batches=50, us=200)

dut_params = {
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 32,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
    "DATA_IN_0_PARALLELISM_DIM_0": 1,
    "DATA_IN_0_PARALLELISM_DIM_1": 1,
    "DATA_EXP_0_PRECISION_0": 8,
    "DATA_EXP_0_PRECISION_1": 4,
    "DATA_OUT_0_PRECISION_1": 6,
}


def get_fixed_softmax_config(kwargs={}):
    config = dut_params
    config.update(kwargs)
    return config


torch.manual_seed(1)
CONSTANT_MULT = 0.19
@pytest.mark.dev
def test_fixed_softmax_smoke():
    """
    Some quick tests to check if the module is working.
    """
    path = Path(__file__).parents[1] / "rtl"
    generate_memory.generate_sv_lut(
        "exp",
        dut_params["DATA_IN_0_PRECISION_0"],
        dut_params["DATA_IN_0_PRECISION_1"],
        dut_params["DATA_EXP_0_PRECISION_0"],
        dut_params["DATA_EXP_0_PRECISION_1"],
        path=path,
        constant_mult=CONSTANT_MULT,
        floor=True,
    )
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_softmax_config(),
        ],
        # skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_softmax_smoke()
