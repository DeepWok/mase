#!/usr/bin/env python3

import os, logging
from random import randint
from pathlib import Path

import torch

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver, sign_extend_t, sign_extend, signed_to_unsigned
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    ErrorThresholdStreamMonitor,
)

from mase_components.common.test.comparator_tree_tb import ComparatorTreeTB
from mase_components.activations.test.softermax_lpw_pow2_tb import LPW_Pow2TB

import cocotb
from cocotb.triggers import *

logger = logging.getLogger("testbench")
logger.setLevel("DEBUG")


class SoftermaxLocalWindowTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "PARALLELISM",
            "IN_WIDTH", "IN_FRAC_WIDTH", "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "SUBTRACT_WIDTH", "SUBTRACT_FRAC_WIDTH"
        ])

        # Driver/Monitor
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk, (dut.out_values, dut.out_max), dut.out_valid, dut.out_ready, check=True
        )

    def generate_inputs(self, batches=10):
        return [[randint(0, 2**self.IN_WIDTH-1) for _ in range(self.PARALLELISM)]
                for _ in range(batches)]

    def _lpw_pow2_model(self, inputs):
        """
        Copy over model for lpw_pow2 since verilator does not allow acessing
        signals/modules which are in generate block in cocotb.
        https://github.com/cocotb/cocotb/issues/1884
        """
        in_t = torch.tensor(inputs)
        num = sign_extend_t(in_t, self.SUBTRACT_WIDTH) / (2 ** self.SUBTRACT_FRAC_WIDTH)
        res = 2 ** num
        res = (res * 2**self.OUT_FRAC_WIDTH).int()
        res = torch.clamp(res, 0, 2**self.OUT_WIDTH-1)
        return res

    def _comparator_tree_model(self, inputs):
        inputs = [[sign_extend(x, self.IN_WIDTH) for x in l] for l in inputs]
        exp_out = [max(l) for l in inputs]
        exp_out = [signed_to_unsigned(x, self.IN_WIDTH) for x in exp_out]
        return exp_out

    def model(self, inputs):
        exp_max = self._comparator_tree_model(inputs)
        subtracted_vals = []
        for m, l in zip(exp_max, inputs):
            subtracted_vals.append(
                [sign_extend(x, self.IN_WIDTH) - sign_extend(m, self.IN_WIDTH)
                 for x in l]
            )
        exp_vals = self._lpw_pow2_model(subtracted_vals).tolist()

        # logger.debug("Inputs: %s" % inputs)
        # logger.debug("Max: %s" % exp_max)
        # logger.debug("Inputs (sign ext): %s" % [[sign_extend(x, self.IN_WIDTH) for x in l] for l in inputs])
        # logger.debug("Max (sign ext): %s" % [sign_extend(x, self.IN_WIDTH) for x in exp_max])
        # logger.debug("Values: %s" % exp_vals)

        return list(zip(exp_vals, exp_max))

    async def run_test(self, batches, us):
        inputs = self.generate_inputs(batches)
        exp_out = self.model(inputs)
        self.in_driver.load_driver(inputs)
        self.output_monitor.load_monitor(exp_out)
        await Timer(us, "us")
        assert self.output_monitor.recv_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = SoftermaxLocalWindowTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=3, us=20)


@cocotb.test()
async def stream(dut):
    tb = SoftermaxLocalWindowTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=1000, us=200)


@cocotb.test()
async def backpressure(dut):
    tb = SoftermaxLocalWindowTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    await tb.reset()
    await tb.run_test(batches=100, us=200)


@cocotb.test()
async def valid(dut):
    tb = SoftermaxLocalWindowTB(dut)
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=100, us=200)


@cocotb.test()
async def valid_backpressure(dut):
    tb = SoftermaxLocalWindowTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.in_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=1000, us=200)



if __name__ == "__main__":

    DEFAULT = {
        "PARALLELISM": 4,
        "IN_WIDTH": 8,
        "IN_FRAC_WIDTH": 2,
        "OUT_WIDTH": 8,
        "OUT_FRAC_WIDTH": 7
    }

    def parallelism_cfgs(cfglist: list):
        out = []
        for cfg in cfglist:
            for d in [1, 2, 4, 16]:
                out.append({**cfg, "PARALLELISM": d})
        return out

    cfgs = [DEFAULT]
    cfgs = parallelism_cfgs(cfgs)

    mase_runner(
        module_param_list=cfgs,
        trace=True,
    )
