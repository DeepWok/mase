#!/usr/bin/env python3

# This script tests the fixed point dot product
import os, logging
from random import randint

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor

import cocotb
from cocotb.triggers import *


class FullThroughputFifoTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "DATA_WIDTH", "SIZE", "ADDR_WIDTH"
        ])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data,
                                      dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(dut.clk, dut.out_data,
                                            dut.out_valid, dut.out_ready)

    def generate_inputs(self, num=20):
        return [randint(0, (2**self.DATA_WIDTH)-1) for _ in range(num)]


@cocotb.test()
async def basic_buffering(dut):
    tb = FullThroughputFifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs()
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(1, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def large_buffering(dut):
    tb = FullThroughputFifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=1000)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(100, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_flip(dut):
    tb = FullThroughputFifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_flip_backpressure(dut):
    tb = FullThroughputFifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))

    inputs = tb.generate_inputs(num=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()



if __name__ == "__main__":
    mase_runner(trace=True)
