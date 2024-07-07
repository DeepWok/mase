#!/usr/bin/env python3

import pytest
import os
from random import randint, choice

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import fixed_preprocess_tensor, bit_driver

from chop.nn.quantized.functional import fixed_softermax


class SoftermaxTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.INFO)

        self.assign_self_params(
            [
                "TOTAL_DIM",
                "PARALLELISM",
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "POW2_WIDTH",
                "POW2_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
            ]
        )
        self.depth = self.TOTAL_DIM // self.PARALLELISM

        self.in_data_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )

        self.out_data_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            width=self.OUT_WIDTH,
            signed=False,
            error_bits=1,
            check=True,
            log_error=True,
        )

        # Model
        self.model = partial(
            fixed_softermax,
            q_config={
                "width": self.IN_WIDTH,
                "frac_width": self.IN_FRAC_WIDTH,
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        # self.in_data_driver.log.setLevel(logging.DEBUG)
        # self.out_data_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batches):
        return torch.randn(
            (batches, self.TOTAL_DIM),
        )

    async def run_test(self, batches, us):
        await self.reset()
        self.log.debug(f"Reset finished")

        inputs = self.generate_inputs(batches)

        for batch in inputs:

            exp_out = self.model(batch)

            # * Load the inputs driver
            self.log.debug(f"Processing inputs: {batch}")
            driver_input = fixed_preprocess_tensor(
                tensor=batch,
                q_config={
                    "width": self.IN_WIDTH,
                    "frac_width": self.IN_FRAC_WIDTH,
                },
                parallelism=[self.PARALLELISM],
            )
            self.in_data_driver.load_driver(driver_input)

            # * Load the output monitor
            self.log.debug(f"Processing outputs: {exp_out}")
            outs = fixed_preprocess_tensor(
                tensor=exp_out,
                q_config={
                    "width": self.OUT_WIDTH,
                    "frac_width": self.OUT_FRAC_WIDTH,
                },
                parallelism=[self.PARALLELISM],
            )
            self.out_data_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.out_data_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = SoftermaxTB(dut)
    tb.out_data_monitor.ready.value = 1
    await tb.run_test(batches=1, us=10)


@cocotb.test()
async def stream(dut):
    tb = SoftermaxTB(dut)
    tb.out_data_monitor.ready.value = 1
    await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def valid_toggle(dut):
    tb = SoftermaxTB(dut)
    tb.in_data_driver.set_valid_prob(0.5)
    tb.out_data_monitor.ready.value = 1
    await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def valid_backpressure_toggle(dut):
    tb = SoftermaxTB(dut)
    tb.in_data_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(tb.out_data_monitor.ready, dut.clk, 0.5))
    await tb.run_test(batches=1000, us=2000)


def get_fixed_softermax_config(kwargs={}):
    config = {
        "TOTAL_DIM": 20,
        "PARALLELISM": 4,
        "IN_WIDTH": 16,
        "IN_FRAC_WIDTH": 6,
        "POW2_WIDTH": 16,
        "OUT_WIDTH": 16,
        "OUT_FRAC_WIDTH": 6,
    }
    config.update(kwargs)
    return config


def get_random_width():
    width = randint(2, 16)
    frac_width = randint(1, width)
    return width, frac_width


def get_random_softermax_config():
    parallelism = choice([2, 4, 8])
    depth = randint(2, 5)
    in_width, in_frac_width = get_random_width()
    out_width, out_frac_width = get_random_width()
    config = {
        "TOTAL_DIM": parallelism * depth,
        "PARALLELISM": parallelism,
        "IN_WIDTH": in_width,
        "IN_FRAC_WIDTH": in_frac_width,
        "POW2_WIDTH": 16,
        "OUT_WIDTH": out_width,
        "OUT_FRAC_WIDTH": out_frac_width,
    }
    return config


@pytest.mark.dev
def test_fixed_softermax_1d_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_softermax_config(),
            *[get_random_softermax_config() for _ in range(50)],
        ],
        jobs=12,
        # skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_softermax_1d_smoke()
