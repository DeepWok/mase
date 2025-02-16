#!/usr/bin/env python3

import logging
from pathlib import Path
from random import randint
from math import ceil
import json

import torch

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import bit_driver
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


class LPW_Reciprocal2TB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            ["ENTRIES", "IN_WIDTH", "IN_FRAC_WIDTH", "OUT_WIDTH", "OUT_FRAC_WIDTH"]
        )

        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)

        # Specify Error Threshold
        self.percentage_error = 0.05  # 5%
        self.error_threshold_bits = ceil(self.percentage_error * (2**self.OUT_WIDTH))

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            width=self.OUT_WIDTH,
            log_error=True,
            signed=False,
            error_bits=self.error_threshold_bits,
            check=False,  # We manually assert later
        )

    def generate_inputs(self, batches=100):
        return [randint(0, 2**self.IN_WIDTH - 1) for _ in range(batches)]

    def sweep_input(self):
        return list(range(2**self.IN_WIDTH))

    def model(self, inputs):
        in_t = torch.tensor(inputs) / (2**self.IN_FRAC_WIDTH)
        recip = 1.0 / in_t
        res = torch.floor(recip * 2**self.OUT_FRAC_WIDTH)
        res = torch.nan_to_num(res)
        res = torch.clamp(res, 0, 2**self.OUT_WIDTH - 1)
        res = res.int()
        return res.tolist()

    async def run_test(self, batches, us):
        await self.reset()
        inputs = self.generate_inputs(batches=batches)
        exp_out = self.model(inputs)
        self.in_driver.load_driver(inputs)
        self.output_monitor.load_monitor(exp_out)
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
    tb = LPW_Reciprocal2TB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    if tb.IN_WIDTH > 16:
        logger.warning("Not doing full sweep due to large input bitwidth.")
        return
    else:
        inputs = tb.sweep_input()
    exp_out = tb.model(inputs)
    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)
    await Timer(4000, "us")
    assert tb.output_monitor.exp_queue.empty()

    # Graphing error
    recv_log = tb.output_monitor.recv_log
    assert len(exp_out) == len(recv_log)

    x = torch.tensor(inputs) / (2**tb.IN_FRAC_WIDTH)
    ref = 1.0 / x
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

    # temp
    data = data.iloc[:100]

    data["hardware error"] = data["hardware fixed-point"] - data["software float32"]
    graph_id = f"{tb.IN_WIDTH}_{tb.IN_FRAC_WIDTH}_to_{tb.OUT_WIDTH}_{tb.OUT_FRAC_WIDTH}"

    # data["sw_error"] = (data["software fixed-point"] - data["software float32"]).abs()
    # data["model_error"] = data["hardware fixed-point"] - data["software fixed-point"]

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
            height=300,
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
        Path(__file__).parent
        / f"build/softermax_lpw_reciprocal/curve_error_{graph_id}.png",
        scale_factor=3,
    )

    max_bit_err = max(tb.output_monitor.error_log)
    average_err = sum(tb.output_monitor.error_log) / len(inputs)

    record = {
        "in_width": tb.IN_WIDTH,
        "in_frac_width": tb.IN_FRAC_WIDTH,
        "out_width": tb.OUT_WIDTH,
        "out_frac_width": tb.OUT_FRAC_WIDTH,
        "max_err": max_bit_err,
        "avg_err": average_err,
    }
    filename = f"{graph_id}.json"
    with open(Path(__file__).parent / "results" / "recip" / filename, "w") as f:
        json.dump(record, f, indent=4)

    tb._final_check()


@cocotb.test()
async def backpressure(dut):
    tb = LPW_Reciprocal2TB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.6))
    await tb.run_test(batches=1000, us=400)


@cocotb.test()
async def valid(dut):
    tb = LPW_Reciprocal2TB(dut)
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(batches=1000, us=400)


@cocotb.test()
async def valid_backpressure(dut):
    tb = LPW_Reciprocal2TB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(batches=1000, us=400)


def width_cfgs():
    cfgs = []
    for width in range(2, 16 + 1):
        frac_width = width // 2
        if frac_width < 3:
            entries = 2**frac_width
        else:
            entries = 8
        cfgs.append(
            {
                "ENTRIES": entries,
                "IN_WIDTH": width,
                "IN_FRAC_WIDTH": frac_width,
                "OUT_WIDTH": width,
                "OUT_FRAC_WIDTH": frac_width,
            }
        )
    return cfgs


def random_cfg():
    in_width = randint(4, 30)
    out_width = randint(4, 30)
    return {
        "ENTRIES": 8,
        "IN_WIDTH": in_width,
        "IN_FRAC_WIDTH": randint(3, in_width - 1),
        "OUT_WIDTH": out_width,
        "OUT_FRAC_WIDTH": randint(3, out_width - 1),
    }


def test_width_cfgs():
    cfgs = width_cfgs()
    cfgs = [
        {
            "ENTRIES": 8,
            "IN_WIDTH": 8,
            "IN_FRAC_WIDTH": 4,
            "OUT_WIDTH": 8,
            "OUT_FRAC_WIDTH": 4,
        }
    ]
    mase_runner(
        module_param_list=cfgs,
    )


def test_smoke():
    mase_runner(
        module_param_list=[
            {
                "ENTRIES": 8,
                "IN_WIDTH": 16,
                "IN_FRAC_WIDTH": 8,
                "OUT_WIDTH": 16,
                "OUT_FRAC_WIDTH": 8,
            }
        ],
    )


if __name__ == "__main__":
    # test_width_cfgs()
    test_smoke()
