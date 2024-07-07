#!/usr/bin/env python3

import os, logging, pytest
from random import randint

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor

import cocotb
from cocotb.triggers import *


class FifoTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["DATA_WIDTH", "DEPTH"])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready, unsigned=True
        )

        # self.in_driver.log.setLevel("DEBUG")
        # self.output_monitor.log.setLevel("DEBUG")

    def generate_inputs(self, num=20):
        return [randint(0, (2**self.DATA_WIDTH) - 1) for _ in range(num)]


@cocotb.test()
async def cocotb_test_basic_buffering(dut):
    tb = FifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=10)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(2, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_large_buffering(dut):
    tb = FifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=100)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(100, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_valid(dut):
    tb = FifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(100, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_backpressure(dut):
    tb = FifoTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.9))

    inputs = tb.generate_inputs(num=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(100, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_valid_backpressure(dut):
    tb = FifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))

    inputs = tb.generate_inputs(num=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(100, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_soak(dut):
    tb = FifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))

    inputs = tb.generate_inputs(num=20000)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(2000, "us")
    assert tb.output_monitor.exp_queue.empty()


@pytest.mark.dev
def test_fifo():
    mase_runner(
        module_param_list=[
            {"DEPTH": 1},
            {"DEPTH": 7},
            {"DEPTH": 8},
            {"DEPTH": 81},
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_fifo()
