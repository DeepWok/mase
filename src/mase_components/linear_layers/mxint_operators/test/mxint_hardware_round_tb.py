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
from utils import mxint_quantize

import torch
from math import ceil, log2
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)

def hardware_round(mx, ex, in_man_frac_width):
    round_max = 2**(8-1) - 1
    round_min = -2**(8-1)
    round_x = mx.reshape(-1) // 2**((in_man_frac_width-ex).reshape(-1))
    return torch.clamp(round_x, round_min, round_max)

class MXIntHardwareRoundTB(Testbench):
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

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

        self.input_drivers = {
            "a": self.data_in_0_driver,
        }
        self.output_drivers = {
            "out": self.data_out_0_monitor,
        }
    def generate_inputs(self):
        inputs = []
        exp_outputs = []
        for _ in range(self.num):
            def hardware_round(mx, ex, in_man_frac_width):
                round_max = 2**(8-1) - 1
                round_min = -2**(8-1)
                round_x = mx.reshape(-1) // 2**((in_man_frac_width-ex).reshape(-1))
                print(mx.reshape(-1))
                print((in_man_frac_width-ex).reshape(-1))
                return torch.clamp(round_x, round_min, round_max)
            data = 49 * torch.rand(int(self.dut.BLOCK_SIZE)) - 24.5
            (data_in, mdata_in, edata_in) = mxint_quantize(
                data,
                int(self.dut.DATA_IN_MAN_WIDTH),
                int(self.dut.DATA_IN_EXP_WIDTH),
            )
            n = hardware_round(mdata_in, edata_in, int(self.dut.DATA_IN_MAN_FRAC_WIDTH))
            print(n)
            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            exp_outputs.append(n.int().tolist())
        return inputs, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, exp_outputs = self.generate_inputs()

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MXIntHardwareRoundTB(dut, num=20)
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
            print(
                "shift_result = ", [x.signed_integer for x in dut.shift_result.value]
            )
            print(
                "clamped_n = ", [x.signed_integer for x in dut.clamped_n.value]
            )
        # print(
        #     "data_out_0 = ",
        #     [x.signed_integer for x in dut.data_out_0.value],
        # )
        print("end")
if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_MAN_WIDTH": 8,
                "DATA_IN_MAN_FRAC_WIDTH": 6,
                "DATA_IN_EXP_WIDTH": 4,
                "BLOCK_SIZE": 4,
                "DATA_OUT_WIDTH": 8,
            },
        ],
        sim="questa",
    )
