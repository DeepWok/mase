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
from utils import mxint_quantize

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MXINTVectorMultTB(Testbench):
    def __init__(self, dut, num) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
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
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        inputs = []
        exp_outputs = []
        for _ in range(self.num):
            data = 20 * torch.rand(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = mxint_quantize(
                data,
                int(self.dut.IN_MAN_WIDTH),
                int(self.dut.IN_EXP_WIDTH),
            )
            exp_out, mexp_out, eexp_out = mxint_quantize(
                data_in,
                int(self.dut.OUT_MAN_WIDTH),
                int(self.dut.OUT_EXP_WIDTH),
            )

            eexp_out = eexp_out.int().item()
            mask = 2 ** (int(self.dut.OUT_EXP_WIDTH) - 1)
            eexp_out = (eexp_out & ~mask) - (eexp_out & mask)

            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            exp_outputs.append(([int(mexp) for mexp in mexp_out.tolist()], eexp_out))
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
    tb = MXINTVectorMultTB(dut, num=10)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            # {
            #     "IN_MAN_WIDTH": 6,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 12,
            #     "OUT_EXP_WIDTH": 4,
            #     "BLOCK_SIZE": 4,
            # },
            # {
            #     "IN_MAN_WIDTH": 8,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 3,
            #     "BLOCK_SIZE": 4,
            # },
            {
                "IN_MAN_WIDTH": 8,
                "IN_EXP_WIDTH": 4,
                "OUT_MAN_WIDTH": 49,
                "OUT_EXP_WIDTH": 5,
                "BLOCK_SIZE": 4,
            },
            # {
            #     "IN_MAN_WIDTH": 12,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 4,
            #     "BLOCK_SIZE": 4,
            # },
        ],
    )
