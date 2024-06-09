import logging
import random

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, batched


logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class SingleElementRepeatTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["DATA_WIDTH", "REPEAT"])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready
        )

    def generate_inputs(self, num=10):
        return [random.randint(0, 2**self.DATA_WIDTH - 1) for _ in range(num)]

    def model(self, inputs):
        exp_out = []
        for x in inputs:
            exp_out.extend([x for _ in range(self.REPEAT)])
        return exp_out

    async def run_test(self, batches, us):
        await self.reset()
        inputs = self.generate_inputs(num=batches)
        exp_out = self.model(inputs)
        self.in_driver.load_driver(inputs)
        self.output_monitor.load_monitor(exp_out)
        await Timer(us, units="us")
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = SingleElementRepeatTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=2, us=1)


@cocotb.test()
async def stream(dut):
    tb = SingleElementRepeatTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=4000, us=1000)


@cocotb.test()
async def backpressure(dut):
    tb = SingleElementRepeatTB(dut)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.run_test(batches=1000, us=1000)


@cocotb.test()
async def valid_backpressure(dut):
    tb = SingleElementRepeatTB(dut)
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.run_test(batches=1000, us=1500)


@cocotb.test()
async def valid_backpressure_more_in(dut):
    tb = SingleElementRepeatTB(dut)
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.3))
    await tb.run_test(batches=1000, us=1500)


@cocotb.test()
async def valid_backpressure_more_out(dut):
    tb = SingleElementRepeatTB(dut)
    tb.in_driver.set_valid_prob(0.3)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))
    await tb.run_test(batches=1000, us=1500)


if __name__ == "__main__":

    def generate_random_params():
        return {
            "DATA_WIDTH": random.randint(1, 32),
            "REPEAT": random.randint(2, 6),
        }

    cfgs = [
        {"DATA_WIDTH": 8, "REPEAT": 4},
        *[generate_random_params() for _ in range(16)],
    ]

    mase_runner(
        module_param_list=cfgs,
        trace=True,
        jobs=8,
    )
