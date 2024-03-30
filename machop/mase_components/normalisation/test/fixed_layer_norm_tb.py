#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
from functools import partial
import numpy as np

import sys

# TODO: Remove these.
sys.path.insert(0, "/home/sv720/mase_fork/mase_group7/machop")
sys.path.insert(0, "/home/jlsand/mase_group7/machop")

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *
from cocotb.binary import BinaryValue

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    StreamMonitorRange,
)
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer
from chop.passes.graph.transforms.quantize.quantized_modules import LayerNormInteger

import torch
from torch.nn import LayerNorm, GroupNorm
from queue import Queue


logger = logging.getLogger("tb_signals")
logger.setLevel(logging.DEBUG)


class LayerNormTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(
            dut, dut.clk, dut.rst
        )  # needed to add rst signal for inheritance

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )

        self.bias_driver = StreamDriver(
            dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        )

        self.config = {
            "data_in_width": 8,
            "data_in_frac_width": 3,
            "weight_width": 8,
            "weight_frac_width": 3,
            "bias_width": 8,
            "bias_frac_width": 3,
        }

        self.data_out_0_monitor = StreamMonitorRange(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            self.config["data_in_width"],
            self.config["data_in_frac_width"],
            check=True,
        )


    def preprocess_tensor(self, tensor, quantizer, frac_width, parallelism):
        tensor = quantizer(tensor)
        tensor = (tensor * (2**frac_width)).int()
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    def get_test_case(self, type):
        if type == 'Q_LAYERNORM':
            model = LayerNormInteger(
                16, elementwise_affine=True, config=self.config
            )
            inputs = torch.randn((1, *model.normalized_shape)) * 2.0 + 0.5
        elif type == 'LAYERNORM':
            model = LayerNorm(
                16, elementwise_affine=True
            )
            inputs = torch.randn((1, *model.normalized_shape)) * 2.0 + 0.5
        elif type == 'GROUPNORM':
            model = GroupNorm(
                2, 16
            )
            inputs = torch.randn((1, 16)) * 2.0 + 0.5
        
        data_width = self.config["data_in_width"]
        data_frac_width = self.config["data_in_frac_width"]
        parallelism = int(self.dut.DATA_IN_0_PARALLELISM_DIM_0) * int(
            self.dut.DATA_IN_0_PARALLELISM_DIM_1
        )

        model.reset_parameters()
        print(model.weight)
        print(model.bias)
        model.training = False

        quantizer = partial(
            integer_quantizer, width=data_width, frac_width=data_frac_width
        )

        print("Inputs: ", inputs)
        quantized_inputs = quantizer(inputs)
        print("Inputs (quantized): ", quantized_inputs)

        exp_outputs_model = model(inputs)
        print("Expected outputs model: ", exp_outputs_model)
        print("Expected sum of quantized inputs: ",  quantizer(sum(quantized_inputs[0])))
        print("Expected mean of quantized inputs: ", quantizer(quantizer(sum(quantized_inputs[0])) / len(inputs[0])),)
        print("Expected var of quantized inputs: ", quantizer(quantized_inputs[0].var()))

        inputs = self.preprocess_tensor(inputs, quantizer, data_frac_width, parallelism)
        print("Pre-processed inputs: ", inputs)

        weight = self.preprocess_tensor(
            model.weight, quantizer, data_frac_width, int(parallelism)
        )

        bias = self.preprocess_tensor(
            model.bias, quantizer, data_frac_width, int(parallelism)
        )
        
        exp_outputs_model = quantizer(exp_outputs_model)
        # print("Exp. outputs model: ", exp_outputs_model)

        # # The output from the module is treated as positive integers
        # # convert the negative expected outputs to their positive
        # # equivalent when not treated as 2's complement.
        def convert(x):
            if x >= 0:
                return x
            else:
                new_x = x + (2 ** (data_width - data_frac_width))
                print(x, " ", new_x)
                return new_x

        exp_outputs_model.detach().apply_(convert)
        # exp_outputs_model = exp_outputs_model.floor()
        # print("Exp. outputs model: ", exp_outputs_model)

        exp_outputs_model = (exp_outputs_model * 2**data_frac_width).int()
        exp_outputs_model = exp_outputs_model.reshape(-1, int(parallelism)).tolist()
        return inputs, weight, bias, exp_outputs_model

    async def run_test_case(self, inputs, weight, bias, exp_outputs_model):
        self.data_out_0_monitor.ready.value = 1

        self.data_in_0_driver.load_driver(inputs)
        self.weight_driver.load_driver(weight)
        self.bias_driver.load_driver(bias)

        self.data_out_0_monitor.load_monitor(exp_outputs_model)

        await Timer(800, units="ns")

    async def run_test(self):
        await self.reset()
        if self.dut.NUM_NORMALIZATION_ZONES == 2: 
            await self.run_test_case(*self.get_test_case('GROUPNORM'))
        else:
            await self.run_test_case(*self.get_test_case('Q_LAYERNORM'))
            await self.run_test_case(*self.get_test_case('LAYERNORM'))




@cocotb.test()
async def simple_test(dut):
    tb = LayerNormTB(dut)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 3,
                "DATA_IN_0_PARALLELISM_DIM_0": 16,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "NUM_NORMALIZATION_ZONES": 2, 
            },
            {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 3,
                "DATA_IN_0_PARALLELISM_DIM_0": 16,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
            },
        ],
    )
