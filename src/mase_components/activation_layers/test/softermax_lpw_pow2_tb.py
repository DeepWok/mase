#!/usr/bin/env python3

import os, logging
import json
from random import randint
from pathlib import Path
from math import ceil

import torch

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver, sign_extend_t
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    ErrorThresholdStreamMonitor,
)

import cocotb
from cocotb.triggers import *

import pandas as pd
import altair as alt

logger = logging.getLogger("testbench")
logger.setLevel("DEBUG")


class LPW_Pow2TB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            ["IN_WIDTH", "IN_FRAC_WIDTH", "OUT_WIDTH", "OUT_FRAC_WIDTH"]
        )

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)

        # 0.1% bit error
        self.error_threshold_bits = ceil((2**self.IN_WIDTH) * 0.001)

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            width=self.OUT_WIDTH,
            log_error=True,
            signed=False,
            error_bits=self.error_threshold_bits,
            check=False,
        )

    def sweep_inputs(self):
        negative_nums = torch.arange(
            start=2 ** (self.IN_WIDTH - 1), end=2**self.IN_WIDTH, dtype=torch.int32
        )
        zero_to_one = torch.arange(
            start=0, end=2**self.IN_FRAC_WIDTH, dtype=torch.int32  # one
        )
        return torch.cat((negative_nums, zero_to_one)).tolist()

    def generate_inputs(self, batches=1):
        half = batches // 2
        negative_nums, zero_to_one_nums = half, batches - half
        inputs = torch.concat(
            [
                # Negative Numbers
                torch.randint(
                    low=2 ** (self.IN_WIDTH - 1),
                    high=2**self.IN_WIDTH,
                    size=(negative_nums,),
                    dtype=torch.int32,
                ),
                # Numbers between 0 and 1
                torch.randint(
                    low=0,
                    high=2**self.IN_FRAC_WIDTH,
                    size=(zero_to_one_nums,),
                    dtype=torch.int32,
                ),
            ]
        )
        return inputs.tolist()

    def model(self, inputs):
        in_t = torch.tensor(inputs)
        num = sign_extend_t(in_t, self.IN_WIDTH) / (2**self.IN_FRAC_WIDTH)
        res = 2**num
        res = (res * 2**self.OUT_FRAC_WIDTH).int()
        res = torch.clamp(res, 0, 2**self.OUT_WIDTH - 1)
        return res.tolist()

    async def run_test(self, batches, us):
        await self.reset()
        inputs = self.generate_inputs(batches)
        exp_out = self.model(inputs)
        self.in_driver.load_driver(inputs)
        self.output_monitor.load_monitor(exp_out)
        logger.info("Waiting %d us..." % us)
        await Timer(us, "us")
        assert self.output_monitor.exp_queue.empty()
        self._final_check()

    def _final_check(self):
        max_bit_err = max(self.output_monitor.error_log)
        logger.info("Maximum bit-error: %d", max_bit_err)
        if max_bit_err > self.error_threshold_bits:
            assert False, (
                "Test failed due to high approximation error. Got %d bits of error!"
                % max_bit_err
            )


@cocotb.test()
async def sweep(dut):
    tb = LPW_Pow2TB(dut)

    # Skip if too large
    if tb.IN_WIDTH > 16:
        logger.info("Skipping full sweep since input range is too large!")
        return

    tb.output_monitor.ready.value = 1
    await tb.reset()
    inputs = tb.sweep_inputs()
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    ns = ((2**tb.IN_WIDTH) * 1000) // 5
    logger.info("Waiting %d ns..." % ns)
    await Timer(ns, "ns")
    assert tb.output_monitor.exp_queue.empty()

    # Graphing error
    recv_log = tb.output_monitor.recv_log
    assert len(exp_out) == len(recv_log)

    x = sign_extend_t(torch.tensor(inputs), tb.IN_WIDTH) / (2**tb.IN_FRAC_WIDTH)
    ref = 2**x
    ref *= 2**tb.OUT_FRAC_WIDTH  # scale up
    ref = torch.clamp(ref, 0, 2**tb.OUT_WIDTH - 1)

    software_ref = ref / (2**tb.OUT_FRAC_WIDTH)
    software_res = [x / (2**tb.OUT_FRAC_WIDTH) for x in exp_out]
    hardware_res = [x / (2**tb.OUT_FRAC_WIDTH) for x in recv_log]

    data = pd.DataFrame(
        {
            "x": x.tolist(),
            "software float32": software_ref.tolist(),
            "software fixed-point": software_res,
            "hardware fixed-point": hardware_res,
        }
    )
    data["hardware error"] = data["software float32"] - data["hardware fixed-point"]

    graph_id = f"{tb.IN_WIDTH}_{tb.IN_FRAC_WIDTH}_to_{tb.OUT_WIDTH}_{tb.OUT_FRAC_WIDTH}"

    curve_data = data.melt(
        id_vars="x",
        value_vars=["software float32", "software fixed-point", "hardware fixed-point"],
        value_name="Value",
        var_name="Type",
    )
    curve_fig = (
        alt.Chart(curve_data)
        .mark_line()
        .encode(
            x=alt.X("x").title(f"x (Q{tb.IN_WIDTH}.{tb.IN_FRAC_WIDTH} Fixed-point)"),
            y=alt.Y("Value").title(
                f"y (Q{tb.OUT_WIDTH}.{tb.OUT_FRAC_WIDTH} Fixed-point)"
            ),
            color=alt.Color("Type"),
        )
        .properties(
            width=600,
            height=220,
        )
    )

    error_data = data[["x", "hardware error"]]
    error_fig = (
        alt.Chart(error_data)
        .mark_line()
        .encode(
            x=alt.X("x").title(f"x (Q{tb.IN_WIDTH}.{tb.IN_FRAC_WIDTH} Fixed-point)"),
            y=alt.Y("hardware error").title(f"Error"),
        )
        .properties(
            width=600,
            height=100,
        )
    )

    (curve_fig & error_fig).save(
        Path(__file__).parent / f"build/softermax_lpw_pow2/curve_error_{graph_id}.png",
        scale_factor=3,
    )

    max_bit_err = max(tb.output_monitor.error_log)
    average_err = sum(tb.output_monitor.error_log) / len(software_res)

    record = {
        "in_width": tb.IN_WIDTH,
        "in_frac_width": tb.IN_FRAC_WIDTH,
        "out_width": tb.OUT_WIDTH,
        "out_frac_width": tb.OUT_FRAC_WIDTH,
        "max_err": max_bit_err,
        "avg_err": average_err,
    }
    filename = f"{graph_id}.json"
    with open(Path(__file__).parent / "results" / "pow2" / filename, "w") as f:
        json.dump(record, f, indent=4)

    tb._final_check()


@cocotb.test()
async def backpressure(dut):
    tb = LPW_Pow2TB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.6))
    await tb.run_test(batches=2000, us=100)


@cocotb.test()
async def valid(dut):
    tb = LPW_Pow2TB(dut)
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(batches=2000, us=100)


@cocotb.test()
async def valid_backpressure(dut):
    tb = LPW_Pow2TB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(batches=2000, us=100)


def width_cfgs():
    bitwidths = [2, 8, 16]
    cfgs = []
    for in_width in bitwidths:
        for in_frac_width in range(1, in_width):
            for out_width in bitwidths:
                for out_frac_width in range(1, out_width):
                    cfgs.append(
                        {
                            "IN_WIDTH": in_width,
                            "IN_FRAC_WIDTH": in_frac_width,
                            "OUT_WIDTH": out_width,
                            "OUT_FRAC_WIDTH": out_frac_width,
                        }
                    )
    return cfgs


def common_widths():
    cfgs = []
    for width in range(2, 16 + 1):
        frac_width = width // 2
        cfgs.append(
            {
                "IN_WIDTH": width,
                "IN_FRAC_WIDTH": frac_width,
                "OUT_WIDTH": width,
                "OUT_FRAC_WIDTH": frac_width,
            }
        )
    return cfgs


def test_width_configs():
    cfgs = width_cfgs()
    print(f"Running {len(cfgs)} configs...")
    mase_runner(
        module_param_list=cfgs,
        # jobs=12,
    )


def test_common_widths():
    mase_runner(module_param_list=common_widths())


def test_high_width():
    mase_runner(
        module_param_list=[
            {
                "IN_WIDTH": 20,
                "IN_FRAC_WIDTH": 10,
                "OUT_WIDTH": 20,
                "OUT_FRAC_WIDTH": 10,
            }
        ]
    )


def test_smoke():
    mase_runner(
        module_param_list=[
            {
                "IN_WIDTH": 8,
                "IN_FRAC_WIDTH": 4,
                "OUT_WIDTH": 8,
                "OUT_FRAC_WIDTH": 4,
            }
        ]
    )


if __name__ == "__main__":
    # test_common_widths()
    # test_high_width()
    test_smoke()
