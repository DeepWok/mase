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
from utils import mxint_quantize, block_mxint_quant, pack_tensor_to_mx_listed_chunk

from chop.nn.quantizers.mxint import _mxint_quantize

import torch
from math import ceil, log2
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


class MXIntMatrixCat(Testbench):
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

        self.data_in_1_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_1, dut.edata_in_1),
            dut.data_in_1_valid,
            dut.data_in_1_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

    def generate_inputs(self):

        din1 = []
        din2 = []
        exp_outputs = []

        for _ in range(int(self.dut.DATA_IN_0_TENSOR_SIZE_DIM_1)):

            # print(f'TENSOR_SIZE_DIM1: {}')

            d0 = torch.rand(
                int(self.dut.DATA_IN_0_TENSOR_SIZE_DIM_0),
                int(self.dut.DATA_IN_0_TENSOR_SIZE_DIM_1),
            )
            d1 = torch.rand(
                int(self.dut.DATA_IN_1_TENSOR_SIZE_DIM_0),
                int(self.dut.DATA_IN_1_TENSOR_SIZE_DIM_1),
            )

            (data_in_0, mdata_in_0, edata_in_0) = block_mxint_quant(
                d0,
                {
                    "width": int(self.dut.DATA_IN_0_PRECISION_0),
                    "exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
                },
                [
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_1),
                ],
            )

            (data_in_1, mdata_in_1, edata_in_1) = block_mxint_quant(
                d1,
                {
                    "width": int(self.dut.DATA_IN_1_PRECISION_0),
                    "exponent_width": int(self.dut.DATA_IN_1_PRECISION_1),
                },
                [
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_1),
                ],
            )

            din1 += pack_tensor_to_mx_listed_chunk(
                mdata_in_0,
                edata_in_0,
                [
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_1),
                ],
            )
            din2 += pack_tensor_to_mx_listed_chunk(
                mdata_in_1,
                edata_in_1,
                [
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_1),
                ],
            )

        exp_outputs = din1 + din2

        print(f"Din 1: \n {din1}")
        print(f"Din 2: \n {din2}")
        print(f"Dout: \n {exp_outputs}")
        return din1, din2, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, weights, exp_outputs = self.generate_inputs()
        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        self.data_in_1_driver.load_driver(weights)
        # # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = MXIntMatrixCat(dut, num=2)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        module="mxint_matrix_cat",
        trace=True,
        # group="linear_layers/mxint_operators",
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 4,
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 4,
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 2,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
                "DATA_IN_0_PARALLELISM_DIM_0": 2,
                "DATA_IN_0_PARALLELISM_DIM_1": 2,
                "DATA_IN_1_PRECISION_0": 8,
                "DATA_IN_1_PRECISION_1": 4,
                "DATA_IN_1_TENSOR_SIZE_DIM_0": 2,
                "DATA_IN_1_TENSOR_SIZE_DIM_1": 2,
                "DATA_IN_1_PARALLELISM_DIM_0": 2,
                "DATA_IN_1_PARALLELISM_DIM_1": 2,
                "DATA_OUT_0_PRECISION_0": 8,
                "DATA_OUT_0_PRECISION_1": 4,
                "DATA_OUT_0_PARALLELISM_DIM_0": 2,
                "DATA_OUT_0_PARALLELISM_DIM_1": 2,
                "BLOCK_SIZE": 2 * 2,  # Parallelism == Block Size
            },
        ],
    )
