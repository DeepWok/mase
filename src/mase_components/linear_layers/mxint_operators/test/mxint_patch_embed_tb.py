#!/usr/bin/env python3

import os, pytest

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge, ReadOnly

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)
from mase_cocotb.runner import mase_runner

# torch.manual_seed(0)
# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner


class PatchEmbedTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        
        # Data drivers
        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mdata_in_0, dut.edata_in_0), 
            dut.data_in_0_valid, dut.data_in_0_ready
        )
        
        self.cls_token_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mcls_token, dut.ecls_token),
            dut.cls_token_valid, dut.cls_token_ready
        )
        
        self.distill_token_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mdistill_token, dut.edistill_token),
            dut.distill_token_valid, dut.distill_token_ready
        )
        
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mweight, dut.eweight),
            dut.weight_valid, dut.weight_ready
        )
        
        self.input_drivers = {
            "data": self.data_in_0_driver,
            "cls": self.cls_token_driver, 
            "distill": self.distill_token_driver,
            "weight": self.weight_driver,
        }
        
        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_driver = MultiSignalStreamDriver(
                dut.clk, (dut.mbias, dut.ebias),
                dut.bias_valid, dut.bias_ready
            )
            self.input_drivers["bias"] = self.bias_driver
            
        # Output monitor
        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk, (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid, dut.data_out_0_ready, 
            check=False
        )
        self.output_monitors = {"out": self.data_out_0_monitor}

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

    def generate_inputs(self):
        return {
            "data": torch.randn((
                self.get_parameter("IN_X"),
                self.get_parameter("IN_Y"),
                self.get_parameter("IN_C")
            )),
            "cls_token": torch.randn((
                self.get_parameter("OUT_C")
            )),
            "distill_token": torch.randn((
                self.get_parameter("OUT_C")
            )),
            "weight": torch.randn((
                self.get_parameter("OUT_C"),
                self.get_parameter("KERNEL_X") * self.get_parameter("KERNEL_Y") * self.get_parameter("IN_C")
            )),
            "bias": torch.randn((
                self.get_parameter("OUT_C")
            )) if self.get_parameter("HAS_BIAS") == 1 else None
        }

    def generate_exp_outputs(self):
        return torch.randn(
            (self.get_parameter("SLIDING_NUM") + 2, 
             self.get_parameter("OUT_C"))
        )

    async def run_test(self, us, num=1):
        await self.reset()
        self.data_out_0_monitor.ready.value = 1

        for i in range(num):
            # Generate all random inputs
            inputs = self.generate_inputs()
            # Generate random expected outputs instead of using model
            exp_out = self.generate_exp_outputs()
            
            # Process input data
            data_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["data"],
                config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1")
                },
                parallelism=[
                    1,
                    self.get_parameter("IN_C"),
                ]
            )
            self.data_in_0_driver.load_driver(data_inputs)

            # Process cls token
            cls_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["cls_token"],
                config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"), 
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1")
                },
                parallelism=[1,self.get_parameter("UNROLL_OUT_C")]
            )
            self.cls_token_driver.load_driver(cls_inputs)

            # Process distill token
            distill_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["distill_token"],
                config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1")
                },
                parallelism=[1,self.get_parameter("UNROLL_OUT_C")]
            )
            self.distill_token_driver.load_driver(distill_inputs)

            # Process weights
            weight_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["weight"],
                config={
                    "width": self.get_parameter("WEIGHT_PRECISION_0"),
                    "exponent_width": self.get_parameter("WEIGHT_PRECISION_1")
                },
                parallelism=[
                    self.get_parameter("UNROLL_OUT_C"),
                    self.get_parameter("IN_C"),
                ]
            )
            self.weight_driver.load_driver(weight_inputs)

            # Process bias if needed
            if self.get_parameter("HAS_BIAS") == 1:
                bias_inputs = self.preprocess_tensor_for_mxint(
                    tensor=inputs["bias"],
                    config={
                        "width": self.get_parameter("BIAS_PRECISION_0"),
                        "exponent_width": self.get_parameter("BIAS_PRECISION_1")
                    },
                    parallelism=[1,self.get_parameter("UNROLL_OUT_C")]
                )
                self.bias_driver.load_driver(bias_inputs)

            # Load output monitor with random exp_out
            outs = self.preprocess_tensor_for_mxint(
                tensor=exp_out,
                config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
                parallelism=[
                    1,
                    self.get_parameter("UNROLL_OUT_C"), 
                ]
            )
            self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test_patch_embed(dut):
    tb = PatchEmbedTB(dut)
    await tb.run_test(us=10, num=5)

def get_patch_embed_config(kwargs={}):
    """
    Default configuration for patch embedding test
    """
    config = {
        # Basic parameters
        "HAS_BIAS": 1,
        
        # Input dimensions
        "IN_X": 4,  # Input feature map height
        "IN_Y": 4,  # Input feature map width 
        "IN_C": 3,   # Input channels
        
        # Kernel dimensions
        "KERNEL_X": 2,  # Kernel height
        "KERNEL_Y": 2,  # Kernel width
        "OUT_C": 4,    # Output channels
        
        # Parallelism
        "UNROLL_OUT_C": 2,  # Parallel output channels
        
        # Precision configurations
        "DATA_IN_0_PRECISION_0": 8,   # Input mantissa width
        "DATA_IN_0_PRECISION_1": 4,   # Input exponent width
        
        "WEIGHT_PRECISION_0": 8,      # Weight mantissa width
        "WEIGHT_PRECISION_1": 4,      # Weight exponent width
        
        "BIAS_PRECISION_0": 8,        # Bias mantissa width 
        "BIAS_PRECISION_1": 4,        # Bias exponent width
        
        "DATA_OUT_0_PRECISION_0": 10,  # Output mantissa width
        "DATA_OUT_0_PRECISION_1": 4,   # Output exponent width
    }

    # Allow overriding with custom parameters
    config.update(kwargs)
    return config

def test_patch_embed_regression():
    """
    More extensive tests with different parameter configurations
    """
    mase_runner(
        trace=True,
        module_param_list=[
            # Basic test with default config
            get_patch_embed_config(),
            
            # Test with larger dimensions
            # get_patch_embed_config({
            #     "IN_X": 28,
            #     "IN_Y": 28,
            #     "IN_C": 16,
            #     "OUT_C": 128,
            #     "UNROLL_OUT_C": 16
            # }),
            
            # # Test with different precision
            # get_patch_embed_config({
            #     "DATA_IN_0_PRECISION_0": 16,
            #     "DATA_IN_0_PRECISION_1": 6,
            #     "WEIGHT_PRECISION_0": 12,
            #     "WEIGHT_PRECISION_1": 5,
            #     "DATA_OUT_0_PRECISION_0": 16,
            #     "DATA_OUT_0_PRECISION_1": 6
            # })
        ],
        # sim="questa"
    )

if __name__ == "__main__":
    test_patch_embed_regression()  # Use regression test instead of smoke test