#!/usr/bin/env python3

import os
import pytest

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import fixed_preprocess_tensor

from chop.nn.quantized.functional import fixed_softermax


class SoftermaxTB(Testbench):
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
            fixed_softermax,
            q_config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
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
    tb = SoftermaxTB(dut)
    await tb.run_test()


def get_fixed_softermax_config(kwargs={}):
    config = {
        "DATA_IN_0_PRECISION_0": 16,
        "DATA_IN_0_PRECISION_1": 6,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 20,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_OUT_0_PRECISION_0": 16,
        "DATA_OUT_0_PRECISION_1": 6,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": 20,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": 20,
        "DATA_OUT_0_PARALLELISM_DIM_0": 2,
        "DATA_OUT_0_PARALLELISM_DIM_1": 2,
    }
    config.update(kwargs)
    return config


@pytest.mark.dev
def test_fixed_softermax_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_softermax_config(),
        ],
        # skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_softermax_smoke()
