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
from chop.nn.quantizers.integer import _integer_floor_quantize
from typing import Literal, Optional, Tuple, Union, Dict, List
import torch
import math
from functools import partial
import random
from mxint_quant.softmax import MXIntHardwareExp

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)
class MXIntExpTB(Testbench):
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

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        self.input_drivers = {
            "a": self.data_in_0_driver,
        }
        self.output_monitors = {
            "out": self.data_out_0_monitor,
        }
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)
    def generate_inputs(self):
        inputs = []
        expected_outputs = []

        #cx: Test the code with software
        q_config = {
            "data_in_width": int(self.dut.DATA_IN_MAN_WIDTH),
            "data_in_exponent_width": int(self.dut.DATA_IN_EXP_WIDTH),
            "data_r_width": int(self.dut.DATA_R_WIDTH),
            "block_size": int(self.dut.BLOCK_SIZE),
            "data_out_width": int(self.dut.DATA_OUT_MAN_WIDTH),
            "data_out_exponent_width": int(self.dut.DATA_OUT_EXP_WIDTH),
        }

        for _ in range(self.num):
            data = 49 * torch.rand(q_config["block_size"]) - 24.5
            from mxint_quant.quantizers import mxint_quant_block
            from mxint_quant.softmax import MXIntHardwareExp
            (qdata_in, mdata_in, edata_in) = mxint_quant_block(
                data,
                q_config["data_in_width"],
                q_config["data_in_exponent_width"],
            )
            
            module = MXIntHardwareExp(q_config)
            # Calculate expected output using software model
            qout, mout, eout = module(qdata_in)
            
            inputs.append((mdata_in.int().tolist(), int(edata_in.int())))
            expected_outputs.append((mout.reshape(-1).int().tolist(), eout.reshape(-1).int().tolist()))
            
        return inputs, expected_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs, expected_outputs = self.generate_inputs()
        
        self.data_in_0_driver.load_driver(inputs)
        self.data_out_0_monitor.load_monitor(expected_outputs)

        await Timer(500, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    # cocotb.start_soon(check_signal(dut))
    tb = MXIntExpTB(dut, num=20)
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
        print("end")

from mase_components.helper import generate_memory
from pathlib import Path

default_config = {
    "DATA_IN_MAN_WIDTH": 8,
    "DATA_IN_EXP_WIDTH": 4,
    "BLOCK_SIZE": 2,
    "DATA_R_WIDTH": 2,
    "DATA_OUT_MAN_WIDTH": 10,
    "DATA_OUT_EXP_WIDTH": 4,
}
if __name__ == "__main__":
    valid_width = default_config["DATA_R_WIDTH"]
    valid_frac_width = default_config["DATA_R_WIDTH"] - 1

    hash_out_width = default_config["DATA_OUT_MAN_WIDTH"]
    hash_out_frac_width = default_config["DATA_OUT_MAN_WIDTH"] - 2

    generate_memory.generate_sv_lut(
        "power2",
        valid_width,
        valid_frac_width,
        hash_out_width,
        hash_out_frac_width,
        path=Path(__file__).parents[1] / "rtl",
        constant_mult=1,
        floor=False,
    )
    mase_runner(
        trace=True,
        module_param_list=[{
            "DATA_IN_MAN_WIDTH": default_config["DATA_IN_MAN_WIDTH"],
            "DATA_IN_EXP_WIDTH": default_config["DATA_IN_EXP_WIDTH"],
            "BLOCK_SIZE": default_config["BLOCK_SIZE"],
            "DATA_R_WIDTH": default_config["DATA_R_WIDTH"],
            "DATA_OUT_MAN_WIDTH": default_config["DATA_OUT_MAN_WIDTH"],
            "DATA_OUT_EXP_WIDTH": default_config["DATA_OUT_EXP_WIDTH"],
        }],
        # sim="questa",
    )
