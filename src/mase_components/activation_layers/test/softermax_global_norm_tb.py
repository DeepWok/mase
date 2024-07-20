#!/usr/bin/env python3

import logging
from math import ceil

import torch
import numpy as np

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver, sign_extend_t, batched
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor

import cocotb
from cocotb.triggers import *

from chop.nn.quantizers import integer_floor_quantizer
from chop.nn.quantizers.quantizers_for_hw import unsigned_integer_quantizer_for_hw

logger = logging.getLogger("testbench")
logger.setLevel("INFO")


class SoftermaxGlobalNormTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "TOTAL_DIM",
                "PARALLELISM",
                "DEPTH",
                "IN_VALUE_WIDTH",
                "RECIP_WIDTH",
                "RECIP_FRAC_WIDTH",
                "IN_VALUE_FRAC_WIDTH",
                "IN_MAX_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
            ]
        )

        # Driver/Monitor
        self.in_driver = StreamDriver(
            dut.clk, (dut.in_values, dut.in_max), dut.in_valid, dut.in_ready
        )

        # Specify Error Threshold
        self.percentage_error = 0.05  # 5%
        self.error_threshold_bits = ceil(self.percentage_error * (2**self.OUT_WIDTH))

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            width=self.OUT_WIDTH,
            signed=False,
            error_bits=self.error_threshold_bits,
            log_error=True,
            check=True,
        )

    def generate_inputs(self, batches=10):
        # TODO: Take a look at all zero case again
        local_vals = torch.randint(
            1, 2**self.IN_VALUE_WIDTH, size=(batches * self.DEPTH, self.PARALLELISM)
        )
        local_max = torch.randint(
            0, 2**self.IN_MAX_WIDTH, size=(batches * self.DEPTH, 1)
        )

        logger.debug("local_vals: %s" % (local_vals))
        logger.debug(
            "local_vals (float): %s" % (local_vals / (2**self.IN_VALUE_FRAC_WIDTH))
        )
        logger.debug("local_max: %s" % (local_max))
        logger.debug(
            "local_max (signed): %s" % (sign_extend_t(local_max, self.IN_MAX_WIDTH))
        )

        return local_vals.tolist(), local_max.flatten().tolist()

    def model(self, inputs):
        batched_in = list(batched(inputs, self.DEPTH))
        exp_output = []

        for batch in batched_in:
            local_vals, local_max = list(zip(*batch))
            local_vals = torch.tensor(list(local_vals), dtype=torch.float) / (
                2**self.IN_VALUE_FRAC_WIDTH
            )
            local_max = torch.tensor(list(local_max), dtype=torch.float)
            local_max = sign_extend_t(
                torch.tensor(list(local_max), dtype=torch.float), self.IN_MAX_WIDTH
            )

            global_max = local_max.max()
            adj_amt = global_max - local_max.reshape(self.DEPTH, 1)
            adj_values = integer_floor_quantizer(
                x=local_vals / (2**adj_amt),
                width=self.IN_VALUE_WIDTH,
                frac_width=self.IN_VALUE_FRAC_WIDTH,
                is_signed=False,
            )
            norm = adj_values.sum()
            inv_norm = integer_floor_quantizer(
                x=1 / (norm + 1e-10),
                width=self.RECIP_WIDTH,
                frac_width=self.RECIP_FRAC_WIDTH,
                is_signed=False,
            )
            softermax = adj_values * inv_norm
            softermax_int = unsigned_integer_quantizer_for_hw(
                softermax, self.OUT_WIDTH, self.OUT_FRAC_WIDTH
            )

            logger.debug("Values: %s" % (local_vals))
            logger.debug("Max: %s -> %s" % (local_max, global_max))
            logger.debug("Diff: %s" % (adj_amt))
            logger.debug("ADJ Values: %s" % (adj_values))
            logger.debug("norm: %s" % (norm))
            logger.debug("softermax: %s" % (softermax))
            logger.debug("softermax (int): %s" % (softermax_int))
            logger.debug("sanity sum: %s" % (softermax.sum().item()))
            logger.debug("integer sum: %s" % (softermax_int.sum().item()))

            # logger.info(adj_values)
            # logger.info(norm)
            # logger.info(softermax)

            # assert abs(softermax.sum().item() - 1) < 0.1, f"Sum is {softermax.sum().item()}"

            exp_output.append(softermax_int)

        return torch.cat(exp_output, dim=0).tolist()

    async def run_test(self, batches, us):
        inputs = self.generate_inputs(batches)
        driver_inputs = list(zip(*inputs))
        exp_out = self.model(driver_inputs)
        self.in_driver.load_driver(driver_inputs)
        self.output_monitor.load_monitor(exp_out)
        await Timer(us, "us")
        assert self.output_monitor.recv_queue.empty()
        self._final_check()

    def _final_check(self):
        if len(self.output_monitor.error_log) == 0:
            logger.info("No Errors.")
            # No errors
            return
        errors = np.stack(self.output_monitor.error_log)
        max_bit_err = np.max(errors)
        logger.info("Maximum bit-error: %d", max_bit_err)
        if max_bit_err > self.error_threshold_bits:
            assert False, (
                "Test failed due to high approximation error. Got %d bits of error!"
                % max_bit_err
            )


@cocotb.test()
async def basic(dut):
    tb = SoftermaxGlobalNormTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=1, us=2)


@cocotb.test()
async def stream(dut):
    tb = SoftermaxGlobalNormTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def backpressure(dut):
    tb = SoftermaxGlobalNormTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    await tb.reset()
    await tb.run_test(batches=100, us=2000)


@cocotb.test()
async def valid(dut):
    tb = SoftermaxGlobalNormTB(dut)
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=100, us=2000)


@cocotb.test()
async def valid_backpressure(dut):
    tb = SoftermaxGlobalNormTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.in_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=1000, us=2000)


if __name__ == "__main__":

    DEFAULT = {
        "TOTAL_DIM": 16,
        "PARALLELISM": 4,
        "IN_VALUE_WIDTH": 16,
        "IN_MAX_WIDTH": 3,
        "OUT_WIDTH": 8,
        "OUT_FRAC_WIDTH": 7,
    }

    def parallelism_cfgs(cfgs: list):
        new_cfgs = []
        for cfg in cfgs:
            for par in [1, 4, 8]:  # parallelism
                for depth in [2, 4, 8]:
                    total = depth * par
                    new_cfgs.append({**cfg, "TOTAL_DIM": total, "PARALLELISM": par})
        return new_cfgs

    def in_value_cfgs(cfgs: list):
        new_cfgs = []
        for cfg in cfgs:
            for in_width in [4, 7, 10]:
                new_cfgs.append(
                    {
                        **cfg,
                        "IN_VALUE_WIDTH": in_width,
                    }
                )
        return new_cfgs

    def in_max_cfgs(cfgs: list):
        new_cfgs = []
        for cfg in cfgs:
            for in_max in [2, 3, 4]:
                new_cfgs.append(
                    {
                        **cfg,
                        "IN_MAX_WIDTH": in_max,
                    }
                )
        return new_cfgs

    gen_cfgs = parallelism_cfgs([{}])
    gen_cfgs = in_value_cfgs(gen_cfgs)
    gen_cfgs = in_max_cfgs(gen_cfgs)

    cfgs = [
        DEFAULT,
        *gen_cfgs,
    ]

    # cfgs = [{'TOTAL_DIM': 32, 'PARALLELISM': 4, 'IN_VALUE_WIDTH': 16, 'IN_MAX_WIDTH': 2}]
    mase_runner(
        module_param_list=cfgs,
        trace=True,
        jobs=12,
    )
