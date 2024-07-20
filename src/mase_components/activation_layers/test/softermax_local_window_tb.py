#!/usr/bin/env python3

import logging
from random import randint

import torch

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver, sign_extend_t, sign_extend, signed_to_unsigned
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
)

from mase_components.cast.test.fixed_signed_cast_tb import _fixed_signed_cast_model

import cocotb
from cocotb.triggers import *

import pytest

logger = logging.getLogger("testbench")
logger.setLevel("INFO")


class SoftermaxLocalWindowTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "PARALLELISM",
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                "MAX_WIDTH",
                "SUBTRACT_WIDTH",
                "SUBTRACT_FRAC_WIDTH",
            ]
        )

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        self.output_monitor = StreamMonitor(
            dut.clk,
            (dut.out_values, dut.out_max),
            dut.out_valid,
            dut.out_ready,
            check=True,
        )

    def generate_inputs(self, batches=10):
        return [
            [randint(0, 2**self.IN_WIDTH - 1) for _ in range(self.PARALLELISM)]
            for _ in range(batches)
        ]

    # def _lpw_pow2_model(self, inputs):
    #     """
    #     Copy over model for lpw_pow2 since verilator does not allow acessing
    #     signals/modules which are in generate block in cocotb.
    #     https://github.com/cocotb/cocotb/issues/1884
    #     """
    #     in_t = torch.tensor(inputs)
    #     num = sign_extend_t(in_t, self.SUBTRACT_WIDTH) / (2 ** self.SUBTRACT_FRAC_WIDTH)
    #     res = 2 ** num
    #     res = (res * 2**self.OUT_FRAC_WIDTH).int()
    #     res = torch.clamp(res, 0, 2**self.OUT_WIDTH-1)
    #     return res

    # def _comparator_tree_model(self, inputs):
    #     inputs = [[sign_extend(x, self.IN_WIDTH) for x in l] for l in inputs]
    #     exp_out = [max(l) for l in inputs]
    #     exp_out = [signed_to_unsigned(x, self.IN_WIDTH) for x in exp_out]
    #     return exp_out

    def model(self, inputs):
        sign_ext = sign_extend_t(
            torch.tensor(inputs, dtype=torch.float), bits=self.IN_WIDTH
        )
        float_inputs = sign_ext / (2**self.IN_FRAC_WIDTH)
        # float_inputs = torch.tensor([[-31.5, -32]])
        rounded_inputs_float, rounded_inputs_uint = _fixed_signed_cast_model(
            float_inputs, self.MAX_WIDTH, 0, False, "floor"
        )
        local_max = rounded_inputs_float.max(dim=1, keepdim=True).values
        local_max_uint = signed_to_unsigned(local_max.int(), self.MAX_WIDTH)

        difference = float_inputs - local_max
        pow2 = 2**difference
        res = torch.clamp(
            (pow2 * 2**self.OUT_FRAC_WIDTH).int(), 0, 2**self.OUT_WIDTH - 1
        )

        logger.debug("float_inputs: %s" % float_inputs)
        logger.debug("rounded_inputs_float: %s" % rounded_inputs_float)
        logger.debug("local_max: %s" % local_max)
        logger.debug("local_max_uint: %s" % local_max_uint)
        logger.debug("difference: %s" % difference)
        logger.debug("pow2: %s" % pow2)
        logger.debug("res: %s" % res)

        exp_vals = res.tolist()
        exp_max = local_max_uint.tolist()

        logger.debug("exp_vals: %s" % exp_vals)
        logger.debug("exp_max: %s" % exp_max)

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
        "OUT_FRAC_WIDTH": 7,
    }

    def parallelism_cfgs(cfglist: list):
        out = []
        for cfg in cfglist:
            for d in [2, 4, 16]:
                out.append({**cfg, "PARALLELISM": d})
        return out

    cfgs = [DEFAULT]
    cfgs = parallelism_cfgs(cfgs)

    mase_runner(
        module_param_list=cfgs,
        trace=True,
        jobs=4,
    )
