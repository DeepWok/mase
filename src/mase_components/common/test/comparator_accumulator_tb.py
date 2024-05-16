#!/usr/bin/env python3

import os, logging
from random import randint
from pathlib import Path

import torch

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver, sign_extend, signed_to_unsigned, batched
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
)

import cocotb
from cocotb.triggers import *

logger = logging.getLogger("testbench")
logger.setLevel("DEBUG")


class ComparatorAccumulatorTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["DATA_WIDTH", "DEPTH", "MAX1_MIN0", "SIGNED"])

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready, check=False
        )

    def generate_inputs(self, batches=3):
        return [randint(0, 2**self.DATA_WIDTH - 1) for _ in range(self.DEPTH * batches)]

    def model(self, inputs):

        batched_in = batched(inputs, self.DEPTH)

        if self.SIGNED:
            batched_in = [
                [sign_extend(x, self.DATA_WIDTH) for x in l] for l in batched_in
            ]

        exp_out = []
        for l in batched_in:
            if self.MAX1_MIN0:
                exp_out.append(max(l))
            else:
                exp_out.append(min(l))

        if self.SIGNED:
            exp_out = [signed_to_unsigned(x, self.DATA_WIDTH) for x in exp_out]

        return exp_out

    async def run_test(self, batches, us):
        inputs = self.generate_inputs(batches=batches)
        exp_out = self.model(inputs)

        # Log the first batch
        logger.info("First Batch")
        logger.info("Input : %s" % inputs[: self.DEPTH])
        logger.info("Expect: %s" % exp_out[: self.DEPTH])

        self.in_driver.load_driver(inputs)
        self.output_monitor.load_monitor(exp_out)

        await Timer(us, "us")
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = ComparatorAccumulatorTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=10, us=10)


@cocotb.test()
async def stream(dut):
    tb = ComparatorAccumulatorTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=500, us=500)


@cocotb.test()
async def backpressure(dut):
    tb = ComparatorAccumulatorTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    await tb.reset()
    await tb.run_test(batches=100, us=100)


@cocotb.test()
async def valid(dut):
    tb = ComparatorAccumulatorTB(dut)
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.6)
    await tb.reset()
    await tb.run_test(batches=100, us=100)


@cocotb.test()
async def backpressure_valid(dut):
    tb = ComparatorAccumulatorTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.in_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=1000, us=1000)


if __name__ == "__main__":

    def depth_cfgs(cfglist: list):
        out = []
        for cfg in cfglist:
            for d in [1, 4, 7, 13]:
                out.append({**cfg, "DEPTH": d})
        return out

    def width_cfgs(cfglist: list):
        out = []
        for cfg in cfglist:
            for size in [2, 4, 8, 16]:
                out.append({**cfg, "DATA_WIDTH": size})
        return out

    def signed_max_min_cfgs(cfglist: list):
        out = []
        for cfg in cfglist:
            out.append({**cfg, "MAX1_MIN0": 0, "SIGNED": 0})
            out.append({**cfg, "MAX1_MIN0": 0, "SIGNED": 1})
            out.append({**cfg, "MAX1_MIN0": 1, "SIGNED": 0})
            out.append({**cfg, "MAX1_MIN0": 1, "SIGNED": 1})
        return out

    DEFAULT = {
        "DATA_WIDTH": 8,
        "DEPTH": 8,
        "MAX1_MIN0": 1,
        "SIGNED": 0,
    }

    cfgs = [DEFAULT]
    cfgs = width_cfgs(cfgs)
    cfgs = depth_cfgs(cfgs)
    cfgs = signed_max_min_cfgs(cfgs)

    mase_runner(
        module_param_list=cfgs,
        trace=True,
        jobs=12,
    )
