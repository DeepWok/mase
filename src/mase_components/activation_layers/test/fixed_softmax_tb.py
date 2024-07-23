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
                "data_in_div_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
            dim=-1,
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

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.out_data_monitor.ready.value = 1

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

        await Timer(1, units="ms")
        assert self.out_data_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = SoftmaxTB(dut)
    await tb.run_test()


dut_params = {
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 128,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 64,
    "DATA_IN_0_PARALLELISM_DIM_0": 16,
    "DATA_IN_0_PARALLELISM_DIM_1": 16,
    "DATA_EXP_0_PRECISION_0": 8,
    "DATA_EXP_0_PRECISION_1": 4,
    "DATA_OUT_0_PRECISION_1": 6,
}


def get_fixed_softmax_config(kwargs={}):
    config = dut_params
    config.update(kwargs)
    return config


torch.manual_seed(1)


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
