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
from mase_cocotb.utils import bit_driver
from mase_cocotb.runner import mase_runner

from mase_cocotb.z_qlayers import quantize_to_int as q2i


class FixedDivTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.dividend_driver = StreamDriver(
            dut.clk, dut.dividend_data, dut.dividend_data_valid, dut.dividend_data_ready
        )
        self.divisor_driver = StreamDriver(
            dut.clk, dut.divisor_data, dut.divisor_data_valid, dut.divisor_data_ready
        )

        self.quotient_monitor = StreamMonitor(
            dut.clk,
            dut.quotient_data,
            dut.quotient_data_valid,
            dut.quotient_data_ready,
            check=True,
        )

        # Set verbosity of driver and monitor loggers to debug
        self.dividend_driver.log.setLevel(logging.DEBUG)
        self.divisor_driver.log.setLevel(logging.DEBUG)
        self.quotient_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        return torch.randint(
            low=1,
            high=100,
            size=(
                1,
                self.get_parameter("IN_NUM"),
            ),
        )

    async def run_test(self, batches, us):
        await self.reset()
        self.log.info(f"Reset finished")
        for _ in range(batches):
            dividend = self.generate_inputs()
            # * Load the inputs driver
            self.log.info(f"Processing dividend: {dividend}")
            qdividend = dividend
            self.dividend_driver.load_driver(qdividend.tolist())

            divisor = self.generate_inputs()
            qdivisor = divisor
            # breakpoint()
            self.log.info(f"Processing divisor: {divisor}")
            self.divisor_driver.load_driver(qdivisor.tolist())
            safe_divisor = torch.where(divisor == 0, torch.tensor(0.000001), divisor)
            result = dividend // safe_divisor
            qresult = result

            self.log.info(f"Processing outputs: {result}")
            self.quotient_monitor.load_monitor(qresult.tolist())

        await Timer(us, units="us")
        assert self.quotient_monitor.exp_queue.empty()


# @cocotb.test()
# async def single_test(dut):
#     tb = FixedDivTB(dut)
#     tb.quotient_monitor.ready.value = 1
#     await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_test(dut):
#     tb = FixedDivTB(dut)
#     tb.quotient_monitor.ready.value = 1
#     await tb.run_test(batches=100, us=200)


@cocotb.test()
async def repeated_backpressure(dut):
    tb = FixedDivTB(dut)
    tb.dividend_driver.set_valid_prob(0.2)
    tb.divisor_driver.set_valid_prob(0.2)
    # tb.quotient_monitor.ready.value = 1

    cocotb.start_soon(bit_driver(dut.quotient_data_ready, dut.clk, 0.5))
    await tb.run_test(batches=10, us=200)


dut_params = {
    "IN_NUM": 1,
    "DIVIDEND_WIDTH": 13,
    "DIVISOR_WIDTH": 20,
    "QUOTIENT_WIDTH": 9,
}


def get_fixed_div_config(kwargs={}):
    config = dut_params
    config.update(kwargs)
    return config


torch.manual_seed(1)


@pytest.mark.dev
def test_fixed_div_smoke():
    """
    Some quick tests to check if the module is working.
    """

    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_div_config(),
        ],
        # skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_div_smoke()
