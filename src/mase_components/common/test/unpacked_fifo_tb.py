#!/usr/bin/env python3

from random import randint

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor

import cocotb
from cocotb.triggers import *


class UnpackedFifoTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["DATA_WIDTH", "DEPTH", "IN_NUM"])

        # Driver/Monitor
        self.in_driver = StreamDriver(
            dut.clk, dut.data_in, dut.data_in_valid, dut.data_in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk, dut.data_out, dut.data_out_valid, dut.data_out_ready
        )

    def generate_inputs(self, batches=20):
        return [
            [randint(0, (2**self.DATA_WIDTH) - 1) for _ in range(self.IN_NUM)]
            for _ in range(batches)
        ]


@cocotb.test()
async def cocotb_test_basic_buffering(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(batches=10)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(2, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_large_buffering(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(batches=1000)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(40, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_valid(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(batches=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_backpressure(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.data_out_ready, dut.clk, 0.9))

    inputs = tb.generate_inputs(batches=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_valid_backpressure(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_ready, dut.clk, 0.7))

    inputs = tb.generate_inputs(batches=200)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_soak(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_ready, dut.clk, 0.7))

    inputs = tb.generate_inputs(batches=20000)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(inputs)

    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_unpacked_fifo():
    mase_runner(trace=True)


if __name__ == "__main__":
    test_unpacked_fifo()
