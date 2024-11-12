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

class MxIntViTAttentionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # Update drivers/monitors for mxint format
        self.data_in_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready
        )
        
        # Query parameters drivers
        self.query_weight_driver = MultiSignalStreamDriver(
            dut.clk, 
            (dut.mweight_query, dut.eweight_query),
            dut.query_weight_valid,
            dut.query_weight_ready
        )
        
        self.query_bias_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mquery_bias, dut.equery_bias),
            dut.query_bias_valid,
            dut.query_bias_ready
        )
        
        # Key parameters drivers
        self.key_weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mkey_weight, dut.ekey_weight), 
            dut.key_weight_valid,
            dut.key_weight_ready
        )
        
        self.key_bias_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mkey_bias, dut.ekey_bias),
            dut.key_bias_valid,
            dut.key_bias_ready
        )
        
        # Value parameters drivers
        self.value_weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mvalue_weight, dut.evalue_weight),
            dut.value_weight_valid,
            dut.value_weight_ready
        )
        
        self.value_bias_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mvalue_bias, dut.evalue_bias),
            dut.value_bias_valid,
            dut.value_bias_ready
        )
        
        # Projection parameters drivers
        self.proj_weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mproj_weight, dut.eproj_weight),
            dut.proj_weight_valid,
            dut.proj_weight_ready
        )
        
        self.proj_bias_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mproj_bias, dut.eproj_bias),  # No exponent for proj bias
            dut.proj_bias_valid,
            dut.proj_bias_ready
        )

        self.out_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False
        )

        self.input_drivers = {
            "data_in": self.data_in_driver,
            "query_weight": self.query_weight_driver,
            "query_bias": self.query_bias_driver,
            "key_weight": self.key_weight_driver,
            "key_bias": self.key_bias_driver,
            "value_weight": self.value_weight_driver,
            "value_bias": self.value_bias_driver,
            "proj_weight": self.proj_weight_driver,
            "proj_bias": self.proj_bias_driver
        }
        self.output_monitors = {"out": self.out_monitor}
        # Model parameters (moved up for clarity)
        self.num_heads = self.get_parameter("NUM_HEADS")
        self.hidden_size = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.seq_len = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1")
        self.head_size = self.hidden_size // self.num_heads
        
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

        # Generate random tensors for all inputs
        batch_size = self.seq_len
        hidden_size = self.hidden_size
        
        # Input data
        input_data = torch.randn((batch_size, hidden_size))
        
        # Query/Key/Value weights and biases
        qkv_weight_shape = (hidden_size, hidden_size)
        qkv_bias_shape = (hidden_size,)
        
        query_weight = torch.randn(qkv_weight_shape)
        query_bias = torch.randn(qkv_bias_shape)
        key_weight = torch.randn(qkv_weight_shape)
        key_bias = torch.randn(qkv_bias_shape)
        value_weight = torch.randn(qkv_weight_shape)
        value_bias = torch.randn(qkv_bias_shape)
        
        # Projection weights and biases
        proj_weight = torch.randn(qkv_weight_shape)
        proj_bias = torch.randn(qkv_bias_shape)

        # Configuration for different parameter types
        input_config = {
            "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
        }
        weight_config = {
            "width": self.get_parameter("WEIGHT_PRECISION_0"),
            "exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
        }
        bias_config = {
            "width": self.get_parameter("BIAS_PRECISION_0"),
            "exponent_width": self.get_parameter("BIAS_PRECISION_1"),
        }
        proj_config = {
            "width": self.get_parameter("WEIGHT_PROJ_PRECISION_0"),
            "exponent_width": self.get_parameter("WEIGHT_PROJ_PRECISION_1"),
        }

        # Parallelism configurations
        input_parallelism = [
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
        ]
        weight_parallelism = [
            self.get_parameter("WEIGHT_PARALLELISM_DIM_1"),
            self.get_parameter("WEIGHT_PARALLELISM_DIM_0"),
        ]
        bias_parallelism = [
            self.get_parameter("BIAS_PARALLELISM_DIM_1"),
            self.get_parameter("BIAS_PARALLELISM_DIM_0"),
        ]
        proj_parallelism = [
            self.get_parameter("WEIGHT_PROJ_PARALLELISM_DIM_1"),
            self.get_parameter("WEIGHT_PROJ_PARALLELISM_DIM_0"),
        ]

        # Preprocess all inputs
        input_data_processed = self.preprocess_tensor_for_mxint(input_data, input_config, input_parallelism)
        
        query_weight_processed = self.preprocess_tensor_for_mxint(query_weight, weight_config, weight_parallelism)
        query_bias_processed = self.preprocess_tensor_for_mxint(query_bias, bias_config, bias_parallelism)
        
        key_weight_processed = self.preprocess_tensor_for_mxint(key_weight, weight_config, weight_parallelism)
        key_bias_processed = self.preprocess_tensor_for_mxint(key_bias, bias_config, bias_parallelism)
        
        value_weight_processed = self.preprocess_tensor_for_mxint(value_weight, weight_config, weight_parallelism)
        value_bias_processed = self.preprocess_tensor_for_mxint(value_bias, bias_config, bias_parallelism)
        
        proj_weight_processed = self.preprocess_tensor_for_mxint(proj_weight, proj_config, proj_parallelism)
        proj_bias_processed = self.preprocess_tensor_for_mxint(proj_bias, bias_config, bias_parallelism)

        # Load all drivers
        self.data_in_driver.load_driver(input_data_processed)
        
        self.query_weight_driver.load_driver(query_weight_processed)
        self.query_bias_driver.load_driver(query_bias_processed)
        
        self.key_weight_driver.load_driver(key_weight_processed)
        self.key_bias_driver.load_driver(key_bias_processed)
        
        self.value_weight_driver.load_driver(value_weight_processed)
        self.value_bias_driver.load_driver(value_bias_processed)
        
        self.proj_weight_driver.load_driver(proj_weight_processed)
        self.proj_bias_driver.load_driver(proj_bias_processed)
        breakpoint()

        # Generate expected output (for verification)
        exp_out = torch.randn((batch_size, hidden_size))
        out_config = {
            "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
        }
        out_parallelism = [
            self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
            self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
        ]
        out_processed = self.preprocess_tensor_for_mxint(exp_out, out_config, out_parallelism)
        self.out_monitor.load_monitor(out_processed)

        await Timer(1, units="ms")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError("Output monitor is not empty at end of test")


async def check_signal(dut):
    await Timer(40, units="ns")
    cycle_count = 0
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        print(f"\nCycle {cycle_count}:")
        print("Valid/Ready Status:")
        print("=" * 50)
        
        # Print QKV internal signals status with exact binary values
        print("\nQKV Internal Handshaking:")
        print("Query signals:")
        print(f"  joint_query:     valid={int(dut.joint_query_valid.value):1d} ready={int(dut.joint_query_ready.value):1d}")
        print(f"  split_query:     valid={[int(x) for x in dut.split_query_valid.value]} ready={[int(x) for x in dut.split_query_ready.value]}")
        
        print("\nKey signals:")
        print(f"  joint_key:       valid={int(dut.joint_key_valid.value):1d} ready={int(dut.joint_key_ready.value):1d}")
        print(f"  split_key:       valid={[int(x) for x in dut.split_key_valid.value]} ready={[int(x) for x in dut.split_key_ready.value]}")
        
        print("\nValue signals:")
        print(f"  joint_value:     valid={int(dut.joint_value_valid.value):1d} ready={int(dut.joint_value_ready.value):1d}")
        print(f"  split_value:     valid={[int(x) for x in dut.split_value_valid.value]} ready={[int(x) for x in dut.split_value_ready.value]}")
        
        # Print other signals as before
        print("\nOther Signals:")
        print("-" * 50)
        if dut.data_in_0_valid.value and dut.data_in_0_ready.value:
            print("INPUT DATA:")
            print(f"  m={[x.signed_integer for x in dut.mdata_in_0.value]}")
            print(f"  e={dut.edata_in_0.value.signed_integer}")
        
        if dut.data_out_0_valid.value and dut.data_out_0_ready.value:
            print("OUTPUT DATA:")
            print(f"  m={[x.signed_integer for x in dut.mdata_out_0.value]}")
            print(f"  e={dut.edata_out_0.value.signed_integer}")
        
        print("\n" + "=" * 50)
        cycle_count += 1

@cocotb.test()
async def cocotb_test(dut):
    cocotb.start_soon(check_signal(dut))  # Enable signal monitoring
    tb = MxIntViTAttentionTB(dut)
    await tb.run_test()

default_config = {
    # Number of attention heads
    "NUM_HEADS": 2,

    # Input data parameters
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 8,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
    "DATA_IN_0_PARALLELISM_DIM_0": 2,
    "DATA_IN_0_PARALLELISM_DIM_1": 1,
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 3,

    # Weight parameters (shared by Q,K,V)
    "WEIGHT_TENSOR_SIZE_DIM_0": 8,
    "WEIGHT_TENSOR_SIZE_DIM_1": 8,
    "WEIGHT_PARALLELISM_DIM_0": 2,
    "WEIGHT_PARALLELISM_DIM_1": 2,
    "WEIGHT_PRECISION_0": 8,
    "WEIGHT_PRECISION_1": 3,

    # Bias parameters (shared by Q,K,V)
    "HAS_BIAS": 1,
    "BIAS_PRECISION_0": 8,
    "BIAS_PRECISION_1": 3,

    # Internal precision parameters
    "QKV_PRECISION_0": 16,
    "QKV_PRECISION_1": 3,

    # Projection parameters
    "WEIGHT_PROJ_PRECISION_0": 12,
    "WEIGHT_PROJ_PRECISION_1": 3,
    "BIAS_PROJ_PRECISION_0": 8,
    "BIAS_PROJ_PRECISION_1": 3,
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
        # sim="questa",
    )


if __name__ == "__main__":
    test_fixed_self_attention_head_smoke()
