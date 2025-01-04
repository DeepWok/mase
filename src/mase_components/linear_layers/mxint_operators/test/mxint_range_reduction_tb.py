#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    StreamMonitor,
)

from mase_cocotb.runner import mase_runner
from mxint_quant import mxint_quant_block, mxint_hardware
from typing import Literal, Optional, Tuple, Union, Dict, List
import torch
import math
from functools import partial
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)

def quantized_range_reduction(mx, ex, in_man_width, data_out_n_width):
    """Vectorized range reduction"""
    def hardware_round(mx, ex, in_man_frac_width, data_out_width):
        round_max = 2**(data_out_width-1) - 1
        round_min = -2**(data_out_width-1)
        round_x = mx.reshape(-1) // 2**((in_man_frac_width-ex).reshape(-1))
        return torch.clamp(round_x, round_min, round_max)
    coefficient_quant_block = partial(
        mxint_quant_block, 
        width=8,
        exponent_width=4, 
        round_bits=4)
    _, mlog2_e, elog2_e = coefficient_quant_block(torch.log2(torch.tensor(math.e)))
    _, mln_2, eln_2 = coefficient_quant_block(torch.log(torch.tensor(2.0)))
    n = hardware_round(mx * mlog2_e, ex + elog2_e, (in_man_width - 1 + 7), data_out_n_width)
    print(n)
    _mx = n * mln_2
    _ex = eln_2
    shifted_mx = mx // 2**(_ex - ex + (in_man_width - 1) - 7)
    print(shifted_mx)
    print(_ex - ex + (in_man_width - 1) - 7)
    mr = shifted_mx - _mx
    breakpoint()
    # return mr as an fixedpoint ?.7 we can make it 2.7
    # return n as an integer number with width = data_out_width
    return mr, n

class MXIntRangeReductionTB(Testbench):
    def __init__(self, dut, num) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready,
        )

        self.data_out_n_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_n,
            dut.data_out_n_valid,
            dut.data_out_n_ready,
            check=True,
        )

        self.data_out_r_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_r,
            dut.data_out_r_valid,
            dut.data_out_r_ready,
            check=True,
        )

        self.input_drivers = {
            "a": self.data_in_0_driver,
        }
        self.output_monitors = {
            "n": self.data_out_n_monitor,
            "r": self.data_out_r_monitor,
        }

    def generate_inputs(self):
        inputs = []
        exp_r_outputs = []
        exp_n_outputs = []
        for _ in range(self.num):
            torch.manual_seed(0)
            data = 49 * torch.rand(int(self.dut.BLOCK_SIZE)) - 24.5
            (data_in, mdata_in, edata_in) = mxint_quant_block(
                data,
                int(self.dut.DATA_IN_MAN_WIDTH),
                int(self.dut.DATA_IN_EXP_WIDTH),
                4,
            )
            r,n = quantized_range_reduction(mdata_in, edata_in, int(self.dut.DATA_IN_MAN_WIDTH), int(self.dut.DATA_OUT_N_WIDTH))
            inputs.append((mdata_in.int().tolist(), int(edata_in)))
            exp_r_outputs.append(r.int().tolist())
            exp_n_outputs.append(n.int().tolist())
        return inputs, exp_r_outputs, exp_n_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_n_monitor.ready.value = 1
        self.data_out_r_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, exp_r_outputs, exp_n_outputs = self.generate_inputs()

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        # Load the output monitors
        print(exp_n_outputs)
        self.data_out_n_monitor.load_monitor(exp_n_outputs)
        self.data_out_r_monitor.load_monitor(exp_r_outputs)

        await Timer(5, units="us")
        assert self.data_out_n_monitor.exp_queue.empty()
        assert self.data_out_r_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MXIntRangeReductionTB(dut, num=20)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        print(dut.data_in_0_valid.value, dut.data_in_0_ready.value)
        if dut.data_in_0_valid.value == 1 and dut.data_in_0_ready.value == 1:
            print(
                "data_in_0 = ", [x.signed_integer for x in dut.mdata_in_0.value]
            )
        if dut.data_out_n_valid.value == 1 and dut.data_out_n_ready.value == 1:
            print(
                "data_out_n = ", [x.signed_integer for x in dut.data_out_n.value]
            )
        #     "straight_data_out_n = ", [x for x in dut.straight_data_out_n.value]
        # )
        # print(
        #     "mdata_in_0_log2_e = ", [x for x in dut.mdata_in_0_log2_e.value]
        # )
        print("end")

if __name__ == "__main__":
    mase_runner(
    trace=True,
    module_param_list=[
        {
        "DATA_IN_MAN_WIDTH": 8,
        "DATA_IN_EXP_WIDTH": 4,
        "BLOCK_SIZE": 4,
        "DATA_OUT_N_WIDTH": 8,
        },
    ],
    sim="verilator",
    )
