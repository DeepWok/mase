#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import sys

sys.path.insert(0, "/home/sv720/mase_fork/mase_group7/machop")
sys.path.insert(0, "/home/jlsand/mase_group7/machop")

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *
from cocotb.binary import BinaryValue

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules import BatchNorm1dInteger

import torch
from queue import Queue

logger = logging.getLogger("tb_signals")
logger.setLevel(logging.DEBUG)


class BatchNormTB(Testbench):
    def __init__(self, dut, num_features=1) -> None:  # , in_features=4, out_features=4
        super().__init__(
            dut, dut.clk, dut.rst
        )  # needed to add rst signal for inheritance

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.v_in_driver = StreamDriver(
            dut.clk, dut.v_in, dut.v_in_valid, dut.v_in_ready
        )

        # self.weight_driver = StreamDriver(
        #     dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        # )

        # self.bias_driver = StreamDriver(
        #     dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        # )

        self.v_out_monitor = StreamMonitor(
            dut.clk,
            dut.v_out,
            dut.v_out_valid,
            dut.v_out_ready,
            check=True,
        )
        """ 
        self.data_in_width = 8
        self.data_in_frac_width = 4
        self.data_out_width = 8
        self.data_out_frac_width = 4
        self.parallelism = 16
        """

        self.model = BatchNorm1dInteger(
            num_features,
            config={
                "data_in_width": 8,
                "data_in_frac_width": 3,
                "weight_width": 8,
                "weight_frac_width": 3,
                "bias_width": 8,
                "bias_frac_width": 3,
            },
        )

    def fe_model(self, data_in):
        # TODO: implement a functionally equivalent model here
        # TODO: combine with random testing
        return data_in

    def preprocess_tensor(self, tensor, quantizer, frac_width, parallelism):
        tensor = quantizer(tensor)
        tensor = (tensor * 2**frac_width).int()
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    def postprocess_tensor(self, tensor, config):
        tensor = [item * (1.0 / 2.0) ** config["frac_width"] for item in tensor]
        return tensor

    async def run_test(self):
        await self.reset()

        data_frac_width = 3
        self.v_out_monitor.ready.value = 1

        # quantizer = partial(
        #     integer_quantizer, width=data_width, frac_width=data_frac_width
        # )
        # inputs = self.preprocess_tensor([2, 64, 56], quantizer, data_frac_width, 1)
        # outputs = self.preprocess_tensor([1.44, 64, 56], quantizer, data_frac_width, 1)
        # print(inputs)

        await Timer(800, units="ns")


@cocotb.test()
async def simple_test(dut):
    print(f"================= DEBUG: in simple_test ================= \n")
    tb = BatchNormTB(dut)
    print(f"================= DEBUG: initialized tb ================= \n")

    await tb.run_test()
    print(f"================= DEBUG: ran test ================= \n")


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "IN_WIDTH": 8,
                "NUM_ITERATION": 10,  # N.B.: changing this requires changes in .sv state enum
            }
        ],
    )

    # def get_dut_parameters(self): #TODO: discuss need for this function
