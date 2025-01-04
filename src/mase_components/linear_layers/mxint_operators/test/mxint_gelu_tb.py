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

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


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
        
        q_config = {
            "data_in_width": int(self.dut.DATA_IN_0_PRECISION_0),
            "data_in_exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
            "data_in_parallelism": [int(self.dut.DATA_IN_0_PARALLELISM_DIM_1), int(self.dut.DATA_IN_0_PARALLELISM_DIM_0)],
            "hash_out_width": int(self.dut.HASH_OUT_WIDTH),
            "data_out_width": int(self.dut.DATA_OUT_0_PRECISION_0),
            "data_out_exponent_width": int(self.dut.DATA_OUT_0_PRECISION_1),
            "data_out_parallelism": [int(self.dut.DATA_OUT_0_PARALLELISM_DIM_1), int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0)],
        }
        
        for _ in range(self.num):
            data = 10 * torch.rand(int(self.dut.DATA_IN_0_PARALLELISM_DIM_0)) - 5  # Generate data between -3 and 3
            from utils import mxint_quant_block
            from mxint_module import mxint_gelu
            (qdata_in, mdata_in, edata_in) = mxint_quant_block(
                data,
                q_config["data_in_width"],
                q_config["data_in_exponent_width"],
            )
            
            # Calculate expected output using software model
            qout, mout, eout = mxint_gelu(
                qdata_in,
                q_config
            )
            
            inputs.append((mdata_in.int().tolist(), int(edata_in.int())))
            expected_outputs.append((mout.reshape(-1).int().tolist(), int(eout.reshape(-1).int())))
            
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
        print("end")

from mase_components.helper import generate_memory
from pathlib import Path
default_config = {
    "data_in_width": 8,
    "data_in_exponent_width": 8,
    "data_in_parallelism": [1, 4],
    "hash_out_width": 8,
    "data_out_width": 8,
    "data_out_exponent_width": 8,
    "data_out_parallelism": [1, 4],
}
if __name__ == "__main__":
    valid_width = default_config["data_in_width"] + 2
    valid_frac_width = default_config["data_in_width"] - 1

    hash_out_width = default_config["hash_out_width"]
    hash_out_frac_width = hash_out_width - 3

    generate_memory.generate_sv_lut(
        "gelu",
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
            "DATA_IN_0_PRECISION_0": default_config["data_in_width"],
            "DATA_IN_0_PRECISION_1": default_config["data_in_exponent_width"],
            "DATA_IN_0_TENSOR_SIZE_DIM_0": 16,
            "DATA_IN_0_PARALLELISM_DIM_0": default_config["data_in_parallelism"][1],
            "HASH_OUT_WIDTH": default_config["hash_out_width"],
            "DATA_OUT_0_PRECISION_0": default_config["data_out_width"],
            "DATA_OUT_0_PRECISION_1": default_config["data_out_exponent_width"],
        }],
        # sim="questa",
    )
