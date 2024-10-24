#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
)

from mase_cocotb.runner import mase_runner
from utils import mxint_quantize

import torch
from math import ceil, log2
import random
from mase_cocotb.utils import bit_driver

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


class Log2_max_abs_tb(Testbench):
    def __init__(self, dut, num=1) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        cocotb.start_soon(check_signal(dut, self.log))
        self.data_in_0_driver = StreamDriver(
            dut.clk,
            dut.data_in_0,
            dut.data_in_0_valid,
            dut.data_in_0_ready,
        )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        self.input_drivers = {"in": self.data_in_0_driver}
        self.output_monitors = {"out": self.data_out_0_monitor}
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        from math import ceil, log2

        data_in = torch.randint(-20, 20, size=(self.get_parameter("IN_SIZE"),))
        log2_max = ceil(log2((int(data_in.abs().max()) + 1e-6)))
        inputs = [data_in.tolist()]
        outputs = [log2_max]
        return inputs, outputs

    async def run_test(self, samples, us):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        self.data_in_0_driver.valid.value = 0
        for _ in range(samples):
            logger.info(f"generating inputs")
            inputs, exp_outputs = self.generate_inputs()

            # Load the inputs driver
            print(inputs)
            self.data_in_0_driver.load_driver(inputs)
            # Load the output monitor
            self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


async def check_signal(dut, log):
    # await Timer(20, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        if str(dut.data_out_0_valid) == "1" and str(dut.data_out_0_ready) == "1":
            print(dut.or_result.value)
        # print("end")


# @cocotb.test()
# async def test(dut):
#     tb = Log2_max_abs_tb(dut, 1)
#     await tb.run_test(samples=10, us=5)

# @cocotb.test()
# async def single_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = Log2_max_abs_tb(dut, 1)
    tb.data_in_0_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
    await tb.run_test(samples=20, us=200)


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            # {
            #     "DATA_IN_0_PRECISION_0": 8,
            #     "DATA_IN_0_PRECISION_1": 4,
            #     "BLOCK_SIZE": 1,
            #     "IN_DEPTH": 1,
            # },
            # {
            #     "DATA_IN_0_PRECISION_0": 8,
            #     "DATA_IN_0_PRECISION_1": 4,
            #     "BLOCK_SIZE": 4,
            # },
            {
                "IN_WIDTH": 8,
                "IN_SIZE": 16,
            },
            {
                "IN_WIDTH": 8,
                "IN_SIZE": 4,
            },
        ],
        sim="questa",
        # gui=True
    )
