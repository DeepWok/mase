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

        # Driver/Monitor
        self.output_monitor = StreamMonitor(
            dut.clk, dut.data_out, dut.data_out_valid, dut.data_out_ready
        )



@cocotb.test()
async def cocotb_test_backpressure(dut):
    tb = UnpackedFifoTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.data_out_ready, dut.clk, 0.9))

    tb.output_monitor.load_monitor([[0,1,2,3],[4,5,6,7]])

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_unpacked_fifo():
    mase_runner(trace=True)


if __name__ == "__main__":
    test_unpacked_fifo()
