#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge, ReadOnly
from pathlib import Path

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import MultiSignalStreamDriver, MultiSignalStreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import ViTSelfAttentionHeadInteger
from chop.nn.quantizers import integer_quantizer, integer_floor_quantizer

from mase_components.helper import generate_memory

import pytest
import math

import torch
from torch import Tensor
import torch.nn as nn
import math

from typing import Optional, Tuple
from functools import partial

from mase_components.linear_layers.mxint_operators.test.utils import MXIntLinearHardware, MXIntMatmulHardware
from mase_components.linear_layers.mxint_operators.test.mxint_softmax_tb import mxint_softmax

class MxIntViTSelfAttentionHeadTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # * QKV drivers with MxInt format
        self.query_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mquery, dut.equery),
            dut.query_valid,
            dut.query_ready
        )
        self.key_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mkey, dut.ekey),
            dut.key_valid,
            dut.key_ready
        )
        self.value_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mvalue, dut.evalue),
            dut.value_valid,
            dut.value_ready
        )

        self.out_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mout, dut.eout),
            dut.out_valid,
            dut.out_ready,
            check=False,
        )

        self.input_drivers = {"in0": self.query_driver, "in1": self.key_driver, "in2": self.value_driver}
        self.output_monitors = {"out": self.out_monitor}
        # Model parameters
        self.head_size = self.get_parameter("IN_DATA_TENSOR_SIZE_DIM_0")
        self.seq_len = self.get_parameter("IN_DATA_TENSOR_SIZE_DIM_1")
        
        # Configure logging
        # self.query_driver.log.setLevel(logging.DEBUG)
        # self.key_driver.log.setLevel(logging.DEBUG)
        # self.value_driver.log.setLevel(logging.DEBUG)
        # self.out_monitor.log.setLevel(logging.DEBUG)

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

    async def run_test(self):
        await self.reset()
        self.log.info("Reset finished")
        self.out_monitor.ready.value = 1

        # Generate random inputs
        query = torch.randn((self.seq_len, self.head_size))
        key = torch.randn((self.seq_len, self.head_size))
        value = torch.randn((self.seq_len, self.head_size))

        # Process and load inputs
        config = {
            "width": self.get_parameter("IN_DATA_PRECISION_0"),
            "exponent_width": self.get_parameter("IN_DATA_PRECISION_1"),
        }
        parallelism = [
            self.get_parameter("IN_DATA_PARALLELISM_DIM_1"),
            self.get_parameter("IN_DATA_PARALLELISM_DIM_0"),
        ]

        query_inputs = self.preprocess_tensor_for_mxint(query, config, parallelism)
        key_inputs = self.preprocess_tensor_for_mxint(key, config, parallelism)
        value_inputs = self.preprocess_tensor_for_mxint(value, config, parallelism)

        self.query_driver.load_driver(query_inputs)
        self.key_driver.load_driver(key_inputs)
        self.value_driver.load_driver(value_inputs)

        # Generate expected outputs (using random values for this example)
        exp_out = torch.randn((self.seq_len, self.head_size))
        out_config = {
            "width": self.get_parameter("OUT_DATA_PRECISION_0"),
            "exponent_width": self.get_parameter("OUT_DATA_PRECISION_1"),
        }
        out_parallelism = [
            self.get_parameter("OUT_DATA_PARALLELISM_DIM_1"),
            self.get_parameter("OUT_DATA_PARALLELISM_DIM_0"),
        ]
        outs = self.preprocess_tensor_for_mxint(exp_out, out_config, out_parallelism)
        self.out_monitor.load_monitor(outs)

        await Timer(1, units="ms")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError("Output monitor is not empty at end of test")


@cocotb.test()
async def cocotb_test(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MxIntViTSelfAttentionHeadTB(dut)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        # Print all valid/ready signals
        print("\nValid/Ready Signals:")
        # print(f"query: {dut.query_valid.value}/{dut.query_ready.value}")
        # print(f"key: {dut.key_valid.value}/{dut.key_ready.value}")
        print(f"qk: {dut.qk_valid.value}/{dut.qk_ready.value}")
        print(f"query_key_linear_acc: {dut.query_key_linear.acc_data_out_valid.value}/{dut.query_key_linear.acc_data_out_ready.value}")
        print(f"query_key_linear_fifo: {dut.query_key_linear.fifo_data_out_valid.value}/{dut.query_key_linear.fifo_data_out_ready.value}") 
        print(f"query_key_linear_cast_buffer: {dut.query_key_linear.cast_i.buffer_data_for_out_valid.value}/{dut.query_key_linear.cast_i.buffer_data_for_out_ready.value}")
        print(f"log2_max_value: {dut.query_key_linear.cast_i.log2_max_value_valid.value}/{dut.query_key_linear.cast_i.log2_max_value_ready.value}")
        # Print data values when valid and ready
        # if dut.query_valid.value == 1 and dut.query_ready.value == 1:
        #     print("query_mout = ", [x.signed_integer for x in dut.mquery.value])
        #     print("query_eout = ", dut.equery.value.signed_integer)
            
        # if dut.key_valid.value == 1 and dut.key_ready.value == 1:
        #     print("key_mout = ", [x.signed_integer for x in dut.mkey.value])
        #     print("key_eout = ", dut.ekey.value.signed_integer)
                       
        if dut.query_key_linear.cast_i.log2_max_value_valid.value == 1 and dut.query_key_linear.cast_i.log2_max_value_ready.value == 1:
            print("log2_max_value = ", dut.query_key_linear.cast_i.log2_max_value.value.signed_integer)
             
        if dut.query_key_linear.cast_i.buffer_data_for_out_valid.value == 1 and dut.query_key_linear.cast_i.buffer_data_for_out_ready.value == 1:
            print("query_key_linear_cast_buffer_mdata = ", [x.signed_integer for x in dut.query_key_linear.cast_i.mbuffer_data_for_out.value])
            print("query_key_linear_cast_buffer_edata = ", dut.query_key_linear.cast_i.ebuffer_data_for_out.value.signed_integer)

        if dut.query_key_linear.cast_i.buffer_data_for_out_valid.value == 1 and dut.query_key_linear.cast_i.buffer_data_for_out_ready.value == 1:
            print("query_key_linear_cast_buffer_mdata = ", [x.signed_integer for x in dut.query_key_linear.cast_i.mbuffer_data_for_out.value])
            print("query_key_linear_cast_buffer_edata = ", dut.query_key_linear.cast_i.ebuffer_data_for_out.value.signed_integer)

        if dut.query_key_linear.acc_data_out_valid.value == 1 and dut.query_key_linear.acc_data_out_ready.value == 1:
            print("query_key_linear_acc_mdata_out = ", [x.signed_integer for x in dut.query_key_linear.acc_mdata_out.value])
            print("query_key_linear_acc_edata_out = ", dut.query_key_linear.acc_edata_out.value.signed_integer)
            
        if dut.qk_valid.value == 1 and dut.qk_ready.value == 1:
            print("qk_mout = ", [x.signed_integer for x in dut.qk_mout.value])
            print("qk_eout = ", dut.qk_eout.value.signed_integer)

        if dut.query_key_linear.fifo_data_out_valid.value == 1 and dut.query_key_linear.fifo_data_out_ready.value == 1:
            print("query_key_linear_fifo_mdata_out = ", [x.signed_integer for x in dut.query_key_linear.fifo_mdata_out.value])
            print("query_key_linear_fifo_edata_out = ", dut.query_key_linear.fifo_edata_out.value.signed_integer)

        print("---")

default_config = {
    "IN_DATA_TENSOR_SIZE_DIM_0": 4,
    "IN_DATA_TENSOR_SIZE_DIM_1": 12,
    "IN_DATA_PARALLELISM_DIM_0": 4,
    "IN_DATA_PARALLELISM_DIM_1": 1,
    "IN_DATA_PRECISION_0": 8,
    "IN_DATA_PRECISION_1": 4,
    "OUT_DATA_PRECISION_0": 8,
    "OUT_DATA_PRECISION_1": 4,
}
def get_fixed_self_attention_head_config(kwargs={}):
    config = default_config
    config.update(kwargs)
    return config


torch.manual_seed(1)


@pytest.mark.dev
def test_fixed_self_attention_head_smoke():
    """
    Some quick tests to check if the module is working.
    """

    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_self_attention_head_config(),
        ],
        skip_build=False,
        sim="verilator",
    )


if __name__ == "__main__":
    test_fixed_self_attention_head_smoke()
