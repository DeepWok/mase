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
from utils import mxint_quantize, block_mxint_quant, MxIntAccumulator
from typing import Literal, Optional, Tuple, Union, Dict, List
import torch
import math
from functools import partial
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)

def quantized_range_reduction(mx, ex, in_man_width, data_out_n_width):
    """Vectorized range reduction"""
    def hardware_round(mx, ex, in_man_frac_width, data_out_width):
        round_max = 2**(data_out_width-1) - 1
        round_min = -2**(data_out_width-1)
        round_x = mx.reshape(-1) // 2**((in_man_frac_width-ex).reshape(-1))
        return torch.clamp(round_x, round_min, round_max)
    coefficient_quant_block = partial(
        mxint_quantize, 
        width=8,
        exponent_width=4)
    _, mlog2_e, elog2_e = coefficient_quant_block(torch.log2(torch.tensor(math.e)))
    _, mln_2, eln_2 = coefficient_quant_block(torch.log(torch.tensor(2.0)))
    n = hardware_round(mx * mlog2_e, ex + elog2_e, (in_man_width - 1 + 7), data_out_n_width)
    print(n)
    _mx = n * mln_2
    _ex = eln_2
    shifted_mx = mx // 2**(_ex - ex + (in_man_width - 1) - 7)
    print(shifted_mx)
    print(_ex - ex + (in_man_width - 1) - 7)
    mr = shifted_mx - _mx
    # return mr as an fixedpoint ?.7 we can make it 2.7
    # return n as an integer number with width = data_out_width
    return mr, n

def fixed_exp(fr):
    frac_width = 7
    exp = 1*2**(frac_width) + fr + fr**2//2**(frac_width + 1) + fr**3*5//2**(frac_width + 4)
    return exp
    

    
def mxint_softmax(x, q_config):
    # fixed_r, integer_n
    in_man_width = q_config["in_man_width"]
    in_exp_width = q_config["in_exp_width"]
    data_out_n_width = q_config["data_out_n_width"]
    data_out_man_width = q_config["data_out_man_width"]
    data_out_frac_width = data_out_man_width - 1
    data_out_exp_width = q_config["data_out_exp_width"]

    shape = x.shape[0]
    mout = torch.zeros_like(x)
    eout = torch.zeros_like(x)

    list_of_mexps = [] 
    list_of_eexps = [] 
    for i in range(shape):
        _, mx, ex = mxint_quantize(x[i], in_man_width, in_exp_width)
        fixed_r, integer_n = quantized_range_reduction(mx, ex, in_man_width, data_out_n_width)
        # fixed_r will be 2.7 bits, integer_n will be data_out_n_width bits
        mexp = fixed_exp(fixed_r)
        eexp = integer_n
        # currently we got mexp ?.7 bits, integer_n data_out_n_width bits
        list_of_mexps.append(mexp)
        list_of_eexps.append(eexp)
    eexps = torch.stack(list_of_eexps)
    mexps = torch.stack(list_of_mexps)
    m_sum, e_sum = MxIntAccumulator(torch.stack(list_of_mexps), torch.stack(list_of_eexps))
    extended_mexps = mexps * 2**(data_out_frac_width)
    pre_cast_mout = extended_mexps // mexps
    pre_cast_eout = eexps - e_sum
    pre_cast_out = pre_cast_mout * 2**(pre_cast_eout - 7)
    for i in range(shape):
        _, mout[i], eout[i] = mxint_quantize(pre_cast_out[i], data_out_man_width, data_out_exp_width)
    return mout, eout


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
        exp_outputs = []
        torch.manual_seed(0)
        for _ in range(self.num):
            data = 49 * torch.rand(int(self.dut.DATA_IN_0_DIM)) - 24.5
            q_config = {
                "in_man_width": int(self.dut.DATA_IN_0_PRECISION_0),
                "in_exp_width": int(self.dut.DATA_IN_0_PRECISION_1),
                "data_out_n_width": int(self.dut.DATA_OUT_0_PRECISION_1),
                "data_out_man_width": int(self.dut.DATA_OUT_0_PRECISION_0),
                "data_out_exp_width": int(self.dut.DATA_OUT_0_PRECISION_1),
            }
            mout, eout = mxint_softmax(data, q_config)
            shape = data.shape[0]
            for i in range(shape):
                _, mdata_in, edata_in = mxint_quantize(data[i], q_config["in_man_width"], q_config["in_exp_width"])
                inputs.append(([int(mdata_in)], int(edata_in)))
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
    cocotb.start_soon(check_signal(dut))
    tb = MXIntSoftmaxTB(dut, num=20)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        # Print all valid/ready signals
        print("\nValid/Ready Signals:")
        print(f"data_in_0: {dut.data_in_0_valid.value}/{dut.data_in_0_ready.value}")
        print(f"data_out_n: {dut.data_out_n_valid.value}/{dut.data_out_n_ready.value}")
        print(f"data_out_r: {dut.data_out_r_valid.value}/{dut.data_out_r_ready.value}")
        print(f"taylor_exp: {dut.taylor_exp_valid.value}/{dut.taylor_exp_ready.value}")
        print(f"acc: {dut.acc_data_out_valid.value}/{dut.acc_data_out_ready.value}")
        
        # Print data values when valid and ready
        if dut.data_in_0_valid.value == 1 and dut.data_in_0_ready.value == 1:
            print("data_in_0 = ", [x.signed_integer for x in dut.mdata_in_0.value])
        
        if dut.data_out_n_valid.value == 1 and dut.data_out_n_ready.value == 1:
            print("data_out_n = ", [x.signed_integer for x in dut.data_out_n.value])
        
        if dut.data_out_r_valid.value == 1 and dut.data_out_r_ready.value == 1:
            print("data_out_r = ", [x.signed_integer for x in dut.data_out_r.value])
            
        if dut.taylor_exp_valid.value == 1 and dut.taylor_exp_ready.value == 1:
            print("taylor_exp = ", dut.taylor_exp.value.signed_integer)
            
        if dut.acc_data_out_valid.value == 1 and dut.acc_data_out_ready.value == 1:
            print("acc_data_out: mdata =", [x.signed_integer for x in dut.acc_mdata_out.value],
                  "edata =", dut.acc_edata_out.value)
        print("---")

if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            {
            "DATA_IN_0_PRECISION_0": 8,
            "DATA_IN_0_PRECISION_1": 4,
            "DATA_IN_0_DIM": 8,
            "DATA_OUT_0_PRECISION_0": 8,
            "DATA_OUT_0_PRECISION_1": 4,
            },
        ],
        sim="verilator",
    )
