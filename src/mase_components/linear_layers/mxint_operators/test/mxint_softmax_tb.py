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
from typing import Literal, Optional, Tuple, Union, Dict, List
import torch
import math
from functools import partial
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)

class MXIntSoftmaxTB(Testbench):
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
        exp_outputs = []
        torch.manual_seed(0)
        from mxint_quant.softmax import MXIntSoftmax
        from mxint_quant.quantizers import mxint_hardware 
        for _ in range(self.num):
            data = 49 * torch.rand(int(self.dut.DATA_IN_0_DIM)) - 24.5
            q_config = {
                "data_in_width": int(self.dut.DATA_IN_0_PRECISION_0),
                "data_in_exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
                "block_size": int(self.dut.BLOCK_SIZE),
                "data_out_width": int(self.dut.DATA_OUT_0_PRECISION_0),
                "data_out_exponent_width": int(self.dut.DATA_OUT_0_PRECISION_1),
                "data_width": int(self.dut.DATA_IN_0_PRECISION_0),
                "data_exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
                "data_r_width": int(self.dut.DATA_R_WIDTH),
                "exp_sum_underflow_bits": int(self.dut.EXP_SUM_UNDERFLOW_BITS),
                "division_underflow_bits": int(self.dut.DIVISION_UNDERFLOW_BITS),
            }
            qdata_in, mdata_in, edata_in = mxint_hardware(
                data, 
                q_config = {
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                    "round_bits": 4,
                },
                parallelism=[1, 1],
            )

            module = MXIntSoftmax(q_config)
            qout, mout, eout = module(data)
            qout, mout, eout = mxint_hardware(
                qout, 
                q_config = {
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                    "round_bits": 4,
                },
                parallelism=[1, 1],
            )
            
            mdata_in = mdata_in.reshape(-1)
            edata_in = edata_in.reshape(-1)
            mout = mout.reshape(-1)
            eout = eout.reshape(-1)
            shape = mdata_in.shape[0]
            for i in range(shape):
                inputs.append(([int(mdata_in[i])], int(edata_in[i])))
                exp_outputs.append(([int(mout[i])], int(eout[i])))
        return inputs, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")

        logger.info(f"generating inputs")
        inputs, exp_outputs = self.generate_inputs()

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        # Load the output monitors
        self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(20, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    # cocotb.start_soon(check_signal(dut))
    tb = MXIntSoftmaxTB(dut, num=20)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        # Print all valid/ready signals
        print("\nValid/Ready Signals:")
        print(f"data_out_0: {dut.data_out_0_valid.value}/{dut.data_out_0_ready.value}")
        print("---")

from mase_components.helper import generate_memory
from pathlib import Path

default_config = {
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "DATA_IN_0_DIM": 8,
    "DATA_OUT_0_PRECISION_0": 8,
    "DATA_OUT_0_PRECISION_1": 4,
    "DATA_R_WIDTH": 2,
    "EXP_SUM_UNDERFLOW_BITS": 1,
    "DIVISION_UNDERFLOW_BITS": 6,
}
if __name__ == "__main__":
    valid_width = default_config["DATA_R_WIDTH"]
    valid_frac_width = default_config["DATA_R_WIDTH"] - 1
    hash_out_width = default_config["DATA_IN_0_PRECISION_0"]
    hash_out_frac_width = default_config["DATA_IN_0_PRECISION_0"] - 2
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
        module_param_list=[
            default_config,
        ],
        sim="verilator",
    )
