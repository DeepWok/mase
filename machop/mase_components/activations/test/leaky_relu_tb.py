#!/usr/bin/env python3

# This script tests the leaky relu activation function
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class LeakyReLUTB(Testbench):
    def __init__(self, dut, in_features=4, out_features=4) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
       

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False,
        )
        # Model
        self.model = LinearInteger(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            config={
                "data_in_width": 16,
                "data_in_frac_width": 3,
                "weight_width": 16,
                "weight_frac_width": 3,
                "bias_width": 16,
                "bias_frac_width": 3,
            },
        )

    def generate_inputs(self):
        return torch.randn((1, self.model.in_features))

    def preprocess_tensor(self, tensor, quantizer, config, parallelism):
        tensor = quantizer(tensor)
        tensor = (tensor * 2 ** config["frac_width"]).int()
        logger.info(f"Tensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        # Load the inputs driver
        logger.info(f"Processing inputs")
        inputs = self.preprocess_tensor(
            inputs,
            self.model.x_quantizer,
            {"width": self.dut.DATA_IN_0_PRECISION_0, "frac_width": 0},
            int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
        )
        
        self.data_in_0_driver.load_driver(inputs)

        # Load the output monitor
        logger.info(f"Processing outputs: {exp_out}")
        
        outs = self.preprocess_tensor(
            exp_out,
            self.model.x_quantizer,
            {"width": 16, "frac_width": 3},
            int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0),
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(1000, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test_20x20(dut):
    tb = LeakyReLUTB(dut, in_features=20, out_features=20)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": 8,                           
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 1,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_PRECISION_0": 16,
                "DATA_OUT_0_PRECISION_1": 8,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 1,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,              
                "INPLACE": 0,
                "NEGATIVE_SLOPE_PRECISION_0": 8,
                "NEGATIVE_SLOPE_PRECISION_1": 7,
                "NEGATIVE_SLOPE_VALUE": 1,
            },
        ],
    )
