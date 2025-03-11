#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)
from mase_cocotb.runner import mase_runner
# from utils import mxint_quantize
# from utils import MxIntCast

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MxIntCastTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in, dut.edata_in),
            dut.data_in_valid,
            dut.data_in_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out, dut.edata_out),
            dut.data_out_valid,
            dut.data_out_ready,
            check=True,
        )
        self.input_drivers = {"in0": self.data_in_0_driver}
        self.output_monitors = {"out": self.data_out_0_monitor}
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, num):
        inputs = []
        exp_outputs = []
        from a_cx_mxint_quant import mxint_quant_block
        for _ in range(num):
            data = 20 * torch.rand(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = mxint_quant_block(
                data,
                int(self.dut.IN_MAN_WIDTH),
                int(self.dut.IN_EXP_WIDTH),
            )
            (exp_out, mexp_out, eexp_out) = mxint_quant_block(
                data_in,
                int(self.dut.OUT_MAN_WIDTH),
                int(self.dut.OUT_EXP_WIDTH),
                round_bits = int(self.dut.ROUND_BITS),
            )
            # mexp_out, eexp_out = MxIntCast(
            #     mdata_in,
            #     edata_in,
            #     {
            #         "in_width": int(self.dut.IN_MAN_WIDTH),
            #         "in_frac_width": int(self.dut.IN_MAN_FRAC_WIDTH),
            #         "in_exponent_width": int(self.dut.IN_EXP_WIDTH),
            #         "out_width": int(self.dut.OUT_MAN_WIDTH),
            #         "out_exponent_width": int(self.dut.OUT_EXP_WIDTH),
            #     },
            # )
            inputs.append((mdata_in.int().tolist(), int(edata_in)))
            exp_outputs.append((mexp_out.int().tolist(), int(eexp_out)))
        return inputs, exp_outputs

    async def run_test(self, us = 1, num = 10):
        await self.reset()
        logger.info(f"Reset finished")

        logger.info(f"generating inputs")
        inputs, exp_outputs = self.generate_inputs(num)

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)

        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)
        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    # cocotb.start_soon(check_signal(dut))
    tb = MxIntCastTB(dut)
    await tb.run_test(us = 10, num = 50)


async def check_signal(dut):
    num = {"data_out_0": 0, "data_in_0": 0}
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.data_out_valid.value == 1 and dut.data_out_ready.value == 1:
            print(dut.edata_out_full)
        print("end")


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "IN_MAN_WIDTH": 13,
                "IN_MAN_FRAC_WIDTH": 12,
                "IN_EXP_WIDTH": 8,
                "OUT_MAN_WIDTH": 8,
                "OUT_EXP_WIDTH": 8,
                "ROUND_BITS": 2,
                "BLOCK_SIZE": 4,
            },
            # {
            #     "IN_MAN_WIDTH": 8,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 3,
            #     "BLOCK_SIZE": 4,
            # },
            # {
            #     "IN_MAN_WIDTH": 8,
            #     "IN_MAN_FRAC_WIDTH": 7,
            #     "IN_EXP_WIDTH": 4,
            #     "OUT_MAN_WIDTH": 16,
            #     "OUT_EXP_WIDTH": 5,
            #     "BLOCK_SIZE": 1,
            # },
            # {
            #     "IN_MAN_WIDTH": 12,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 4,
            #     "BLOCK_SIZE": 4,
            # },
        ],
        # sim="questa",
        # gui=True
    )
