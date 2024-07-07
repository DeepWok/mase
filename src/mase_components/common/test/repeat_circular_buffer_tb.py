import logging
import random
import pytest

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, batched


logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class CircularBufferTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["DATA_WIDTH", "REPEAT", "SIZE"])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready
        )

    def generate_inputs(self, num=10):
        inputs = []
        for _ in range(num):
            inputs.extend(
                [random.randint(0, 2**self.DATA_WIDTH - 1) for _ in range(self.SIZE)]
            )
        return inputs

    def model(self, inputs):
        x = list(batched(inputs, n=self.SIZE))
        y = []
        for seq in x:
            for _ in range(self.REPEAT):
                y.extend(seq)
        return y


@cocotb.test()
async def basic(dut):
    tb = CircularBufferTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=2)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(10, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_stream(dut):
    tb = CircularBufferTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs(num=100)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_backpressure(dut):
    tb = CircularBufferTB(dut)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.reset()

    inputs = tb.generate_inputs(num=200)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(500, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_valid_backpressure(dut):
    tb = CircularBufferTB(dut)
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.reset()

    inputs = tb.generate_inputs(num=200)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(500, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_valid_backpressure_more_input(dut):
    tb = CircularBufferTB(dut)
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    await tb.reset()

    inputs = tb.generate_inputs(num=200)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(500, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_valid_backpressure_more_output(dut):
    tb = CircularBufferTB(dut)
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.7))
    await tb.reset()

    inputs = tb.generate_inputs(num=200)
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(500, units="us")
    assert tb.output_monitor.exp_queue.empty()


def generate_random_params():
    return {
        "DATA_WIDTH": random.randint(1, 12),
        "REPEAT": random.randint(2, 6),
        "SIZE": random.randint(2, 6),
    }


@pytest.mark.dev
def test_repeat_circular_buffer():
    mase_runner(
        module_param_list=[
            # Power of 2 params
            {"DATA_WIDTH": 8, "REPEAT": 2, "SIZE": 4},
            # Change data width
            {"DATA_WIDTH": 17, "REPEAT": 2, "SIZE": 4},
            # Non power of 2 repeats
            {"DATA_WIDTH": 32, "REPEAT": 3, "SIZE": 4},
            # Non power of 2 buffer size
            {"DATA_WIDTH": 32, "REPEAT": 2, "SIZE": 7},
            # Purely random params
            *[generate_random_params() for _ in range(5)],
        ],
        trace=True,
        jobs=8,
    )


if __name__ == "__main__":
    test_repeat_circular_buffer()
