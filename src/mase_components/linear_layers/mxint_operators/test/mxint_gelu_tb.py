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
from chop.nn.quantizers.integer import _integer_floor_quantize
from typing import Literal, Optional, Tuple, Union, Dict, List
import torch
import math
from functools import partial
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)

    
from utils import MxIntCast

def mxint_gelu(mx, ex, q_config):
    """Vectorized range reduction"""
    in_man_width = q_config["in_width"]
    in_exp_width = q_config["in_exponent_width"]
    out_man_width = q_config["out_width"]
    out_exp_width = q_config["out_exponent_width"]
    # first
    real_x = mx * 2**(ex - in_man_width + 1)
    if real_x >= 3:
        out_mx, out_ex = mx, ex
    elif real_x <= -3:
        out_mx, out_ex = 0, ex
    else:
        quant_out_x = _integer_floor_quantize(torch.nn.GELU()(real_x), out_man_width, out_man_width - 1)
        out_mx = quant_out_x * 2**(out_man_width - 1 - ex)
        out_ex = ex

    return out_mx

def mxint_gelu_block(mx, ex, q_config):
    for i in range(len(mx)):
        mx[i] = mxint_gelu(mx[i], ex, q_config)
    out_mx, out_ex = MxIntCast(mx, ex, 
                                {
                                    **q_config,
                                    "in_frac_width": q_config["in_width"] - 1  # VALID_WIDTH - 3
                                } 
    )
    return out_mx, out_ex

class MXIntGeluTB(Testbench):
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
            check=False,
        )
        self.input_drivers = {
            "a": self.data_in_0_driver,
        }
        self.output_monitors = {
            "out": self.data_out_0_monitor,
        }
    def generate_inputs(self):
        inputs = []
        expected_outputs = []
        
        q_config = {
            "in_width": int(self.dut.DATA_IN_0_PRECISION_0),
            "in_exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
            "out_width": int(self.dut.DATA_OUT_0_PRECISION_0),
            "out_exponent_width": int(self.dut.DATA_OUT_0_PRECISION_1),
        }
        
        for _ in range(self.num):
            data = 6 * torch.rand(int(self.dut.DATA_IN_0_PARALLELISM_DIM_0)) - 3  # Generate data between -3 and 3
            (_, mdata_in, edata_in) = mxint_quantize(
                data,
                q_config["in_width"],
                q_config["in_exponent_width"],
            )
            
            # Calculate expected output using software model
            out_mx, out_ex = mxint_gelu_block(
                mdata_in.clone(), 
                edata_in.clone(), 
                q_config
            )
            
            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            expected_outputs.append((out_mx.int().tolist(), out_ex.int().tolist()))
            
        return inputs, expected_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs, expected_outputs = self.generate_inputs()
        
        self.data_in_0_driver.load_driver(inputs)
        self.data_out_0_monitor.load_monitor(expected_outputs)

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MXIntGeluTB(dut, num=20)
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
        # if dut.data_out_n_valid.value == 1 and dut.data_out_n_ready.value == 1:
        #     print(
        #         "data_out_n = ", [x.signed_integer for x in dut.data_out_n.value]
        #     )
        #     "straight_data_out_n = ", [x for x in dut.straight_data_out_n.value]
        # )
        # print(
        #     "mdata_in_0_log2_e = ", [x for x in dut.mdata_in_0_log2_e.value]
        # )
        print("end")

if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[{
            "DATA_IN_0_PRECISION_0": 6,
            "DATA_IN_0_PRECISION_1": 4,
            "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
            "DATA_IN_0_PARALLELISM_DIM_0": 4,
            "DATA_OUT_0_PRECISION_0": 6,
            "DATA_OUT_0_PRECISION_1": 4,
        }],
        # sim="questa",
    )
