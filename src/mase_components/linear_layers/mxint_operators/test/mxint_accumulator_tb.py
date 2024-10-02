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
from math import ceil, log2
import random
from mase_cocotb.utils import bit_driver

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


class MXIntAccumulatorTB(Testbench):
    def __init__(self, dut, num=1) -> None:
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
        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

    def generate_inputs(self):
        from utils import block_mxint_quant, pack_tensor_to_mx_listed_chunk
        from utils import mxint_quantize
        from math import ceil, log2

        data_in = 20 * torch.rand(
            self.get_parameter("IN_DEPTH"), self.get_parameter("BLOCK_SIZE")
        )
        config = {
            "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
        }
        parallelism = [1, self.get_parameter("BLOCK_SIZE")]
        (qtensor, mtensor, etensor) = block_mxint_quant(data_in, config, parallelism)

        qout, mout, eout = mxint_quantize(
            qtensor.sum(dim=0),
            width=config["width"]
            + 2 ** config["exponent_width"]
            + ceil(log2(self.get_parameter("IN_DEPTH"))),
            exponent_width=config["exponent_width"],
            exponent=int(etensor.min()),
        )

        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        exp_outs = [(mout.int().tolist(), int(eout))]

        return tensor_inputs, exp_outs

    async def run_test(self, samples, us):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for _ in range(samples):
            logger.info(f"generating inputs")
            inputs, exp_outputs = self.generate_inputs()

            # Load the inputs driver
            self.data_in_0_driver.load_driver(inputs)
            # Load the output monitor
            self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


# @cocotb.test()
# async def test(dut):
#     tb = MXIntAccumulatorTB(dut, 1)
#     await tb.run_test(samples=20, us=5)

# @cocotb.test()
# async def single_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = MXIntAccumulatorTB(dut, 1)
    tb.data_in_0_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
    await tb.run_test(samples=20, us=200)


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 4,
                "BLOCK_SIZE": 1,
                "IN_DEPTH": 1,
            },
        ],
    )
