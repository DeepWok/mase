import logging
import random

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner


logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

class CircularBufferTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "DATA_WIDTH", "REPEAT", "SIZE"
        ])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data,
                                      dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(dut.clk, dut.out_data,
                                            dut.out_valid, dut.out_ready)

@cocotb.test()
async def single_buffer(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    tb.output_monitor.ready.value = 1
    input_list = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
    EXP_OUT = input_list * tb.REPEAT

    for d in input_list:
        tb.in_driver.append(d)

    for out in EXP_OUT:
        tb.output_monitor.expect(out)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()

@cocotb.test()
async def multi_buffer(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    tb.output_monitor.ready.value = 1
    input_list_0 = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
    input_list_1 = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
    EXP_OUT_0 = input_list_0 * tb.REPEAT
    EXP_OUT_1 = input_list_1 * tb.REPEAT
    EXP_OUT = EXP_OUT_0 + EXP_OUT_1

    for d in (input_list_0 + input_list_1):
        tb.in_driver.append(d)

    for out in EXP_OUT:
        tb.output_monitor.expect(out)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def random_stream(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    tb.output_monitor.ready.value = 1

    DATA_STREAM = []
    EXP_STREAM = []

    for _ in range(10):
        data = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
        exp = data * tb.REPEAT
        DATA_STREAM.extend(data)
        EXP_STREAM.extend(exp)

    for d in DATA_STREAM:
        tb.in_driver.append(d)

    for out in EXP_STREAM:
        tb.output_monitor.expect(out)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()

async def bit_driver(signal, clk, prob_on):
    assert prob_on >= 0.0 and prob_on <= 1.0, "Probability between 0 & 1"
    while True:
        await RisingEdge(clk)
        x = random.random()
        signal.value = 1 if x < prob_on else 0

@cocotb.test()
async def basic_delay_input(dut):
    tb = CircularBufferTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    data = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
    half = tb.SIZE // 2
    EXP_OUT = data * tb.REPEAT

    for _ in range(2):
        for out in EXP_OUT:
            tb.output_monitor.expect(out)

    for d in data[:half]:
        tb.in_driver.append(d)

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    for d in data[half:]:
        tb.in_driver.append(d)

    for d in data:
        tb.in_driver.append(d)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()

@cocotb.test()
async def basic_backpressure(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))

    data = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
    EXP_OUT = data * tb.REPEAT

    for d in data:
        tb.in_driver.append(d)

    for out in EXP_OUT:
        tb.output_monitor.expect(out)

    await Timer(100, units="us")
    assert tb.output_monitor.exp_queue.empty()

@cocotb.test()
async def random_backpressure(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.2))

    DATA_STREAM = []
    EXP_STREAM = []

    for _ in range(10):
        data = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
        exp = data * tb.REPEAT
        DATA_STREAM.extend(data)
        EXP_STREAM.extend(exp)

    for d in DATA_STREAM:
        tb.in_driver.append(d)

    for out in EXP_STREAM:
        tb.output_monitor.expect(out)

    await Timer(1000, units="us")
    assert tb.output_monitor.exp_queue.empty()

@cocotb.test()
async def random_valid_backpressure(dut):
    tb = CircularBufferTB(dut)

    await tb.reset()
    tb.in_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.6))

    DATA_STREAM = []
    EXP_STREAM = []

    for _ in range(10):
        data = [random.randint(0, 2**tb.DATA_WIDTH-1) for _ in range(tb.SIZE)]
        exp = data * tb.REPEAT
        DATA_STREAM.extend(data)
        EXP_STREAM.extend(exp)

    for d in DATA_STREAM:
        tb.in_driver.append(d)

    for out in EXP_STREAM:
        tb.output_monitor.expect(out)

    await Timer(1000, units="us")
    assert tb.output_monitor.exp_queue.empty()


def generate_random_params():
    return {
        "DATA_WIDTH": random.randint(1, 32),
        "REPEAT": random.randint(2, 12),
        "SIZE": random.randint(2, 12),
    }

if __name__ == "__main__":
    mase_runner(module_param_list=[
        # Power of 2 params
        {"DATA_WIDTH": 8, "REPEAT": 2, "SIZE": 4},
        # Change data width
        {"DATA_WIDTH": 17, "REPEAT": 2, "SIZE": 4},
        # Non power of 2 repeats
        {"DATA_WIDTH": 32, "REPEAT": 3, "SIZE": 4},
        # Non power of 2 buffer size
        {"DATA_WIDTH": 32, "REPEAT": 2, "SIZE": 7},
        # Purely random params
        *[generate_random_params() for _ in range(10)]
    ])
