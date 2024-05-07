#!/usr/bin/env python3

import os

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
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )

        self.out_data_monitor = StreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            check=True,
        )
        # Model
        self.model = partial(
            fixed_softermax,
            q_config={
                "width": self.get_parameter("IN_WIDTH"),
                "frac_width": self.get_parameter("IN_FRAC_WIDTH"),
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        self.in_data_driver.log.setLevel(logging.DEBUG)
        self.out_data_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        return torch.randn((self.get_parameter("TOTAL_DIM"),))

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
                "width": self.get_parameter("IN_WIDTH"),
                "frac_width": self.get_parameter("IN_FRAC_WIDTH"),
            },
            parallelism=[self.get_parameter("PARALLELISM")],
        )
        self.in_data_driver.load_driver(inputs)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        outs = fixed_preprocess_tensor(
            tensor=exp_out,
            q_config={
                "width": self.get_parameter("OUT_WIDTH"),
                "frac_width": self.get_parameter("OUT_FRAC_WIDTH"),
            },
            parallelism=[self.get_parameter("PARALLELISM")],
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
        "TOTAL_DIM": 20,
        "PARALLELISM": 4,
        "IN_WIDTH": 16,
        "IN_FRAC_WIDTH": 6,
        "POW2_WIDTH": 16,
        "POW2_FRAC_WIDTH": 6,
        "OUT_WIDTH": 16,
        "OUT_FRAC_WIDTH": 6,
    }
    config.update(kwargs)
    return config


def test_fixed_softermax_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_softermax_config(),
        ],
        skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_softermax_smoke()
