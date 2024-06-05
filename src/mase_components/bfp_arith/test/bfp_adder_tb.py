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
from mase_cocotb.utils import block_fp_quantize

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class BFPAdderTB(Testbench):
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
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mweight, dut.eweight),
            dut.weight_valid,
            dut.weight_ready,
        )
        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

    def generate_inputs(self):
        inputs = []
        weights = []
        exp_outputs = []
        for _ in range(self.num):
            data = torch.rand(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = block_fp_quantize(
                data,
                int(self.dut.DATA_IN_0_PRECISION_0),
                int(self.dut.DATA_IN_0_PRECISION_1),
            )
            w = torch.rand(int(self.dut.BLOCK_SIZE))
            (weight, mweight, eweight) = block_fp_quantize(
                w,
                int(self.dut.WEIGHT_PRECISION_0),
                int(self.dut.WEIGHT_PRECISION_1),
            )
            max_exp = max(edata_in, eweight)

            (pre_cast_data_in, _, _) = block_fp_quantize(
                data_in,
                int(self.dut.DATA_OUT_0_PRECISION_0),
                int(self.dut.DATA_OUT_0_PRECISION_1),
                max_exp,
            )
            (pre_cast_weight, _, _) = block_fp_quantize(
                weight,
                int(self.dut.DATA_OUT_0_PRECISION_0),
                int(self.dut.DATA_OUT_0_PRECISION_1),
                max_exp,
            )
            exp_out, mexp_out, eexp_out = block_fp_quantize(
                pre_cast_data_in + pre_cast_weight,
                int(self.dut.DATA_OUT_0_PRECISION_0),
                int(self.dut.DATA_OUT_0_PRECISION_1) + 1,
                max_exp + 1,
            )
            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            weights.append((mweight.int().tolist(), eweight.int().tolist()))
            exp_outputs.append((mexp_out.int().tolist(), eexp_out.int().tolist()))
        return inputs, weights, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")

        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, weights, exp_outputs = self.generate_inputs()

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        self.weight_driver.load_driver(weights)

        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)
        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = BFPAdderTB(dut, num=20)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 8,
                "WEIGHT_PRECISION_0": 8,
                "WEIGHT_PRECISION_1": 8,
                "BLOCK_SIZE": 1,
            },
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 8,
                "WEIGHT_PRECISION_0": 8,
                "WEIGHT_PRECISION_1": 8,
                "BLOCK_SIZE": 4,
            },
            # {
            #     "DATA_IN_0_PRECISION_0": 8,
            #     "DATA_IN_0_PRECISION_1": 8,
            #     "WEIGHT_PRECISION_0": 9,
            #     "WEIGHT_PRECISION_1": 7,
            #     "BLOCK_SIZE": 4,
            # },
        ],
    )
