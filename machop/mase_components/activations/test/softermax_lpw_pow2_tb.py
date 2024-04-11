#!/usr/bin/env python3

import os, logging
from random import randint
from pathlib import Path

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
        self.assign_self_params([
            "IN_WIDTH", "IN_FRAC_WIDTH", "OUT_WIDTH", "OUT_FRAC_WIDTH"
        ])

        # Driver/Monitor
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready,
            width=self.OUT_WIDTH,
            log_error=True,
            signed=True,
            error_bits=4,
            check=False,
        )

    def generate_inputs(self):
        return torch.arange(
            2**(self.IN_WIDTH-1),
            2**self.IN_WIDTH,
            dtype=torch.int32
        ).tolist() + [0]

    def model(self, inputs):
        in_t = torch.tensor(inputs)
        num = sign_extend_t(in_t, self.IN_WIDTH) / (2 ** self.IN_FRAC_WIDTH)
        res = 2 ** num
        res = (res * 2**self.OUT_FRAC_WIDTH).int()
        res = torch.clamp(res, 0, 2**self.OUT_WIDTH-1)
        return res.tolist()


@cocotb.test()
async def sweep(dut):
    tb = LPW_Pow2TB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1

    inputs = tb.generate_inputs()
    exp_out = tb.model(inputs)

    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(20, "us")
    assert tb.output_monitor.exp_queue.empty()

    # Graphing error
    recv_log = tb.output_monitor.recv_log
    assert len(exp_out) == len(recv_log)

    x = sign_extend_t(torch.tensor(inputs), tb.IN_WIDTH) / (2 ** tb.IN_FRAC_WIDTH)
    ref = (2 ** x)
    ref *= 2 ** tb.OUT_FRAC_WIDTH  # scale up
    ref = torch.clamp(ref, 0, 2**tb.OUT_WIDTH-1)

    data = pd.DataFrame({
        "x": x.tolist(),
        "reference": ref.tolist(),
        "software": exp_out,
        "hardware": recv_log,
    }).melt(
        id_vars="x",
        value_vars=["reference", "software", "hardware"],
        value_name="Value",
        var_name="Type"
    )

    alt.Chart(data).mark_line().encode(
        x="x",
        y="Value",
        color="Type",
    ).properties(
        width=600,
        height=300,
    ).save(
        Path(__file__).parent / f"error_graph_{tb.IN_WIDTH}.png",
        scale_factor=3,
    )


@cocotb.test()
async def backpressure(dut):
    tb = LPW_Pow2TB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.6))
    await tb.reset()

    inputs = tb.generate_inputs()
    exp_out = tb.model(inputs)

    tb.in_driver.load_driver(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(50, "us")
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(module_param_list=[
            {
                "IN_WIDTH": 8,
                "IN_FRAC_WIDTH": 2,
                "OUT_WIDTH": 8,
                "OUT_FRAC_WIDTH": 7
            }
        ],
        seed=0,
        trace=True,
    )
