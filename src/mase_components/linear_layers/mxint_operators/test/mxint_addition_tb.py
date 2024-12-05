
#!/usr/bin/env python3

import os, pytest
import torch
import logging
from functools import partial

import cocotb
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)
from mase_cocotb.runner import mase_runner

class AdditionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        
        # Data drivers
        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mdata_in_0, dut.edata_in_0), 
            dut.data_in_0_valid, dut.data_in_0_ready
        )
        
        self.data_in_1_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mdata_in_1, dut.edata_in_1),
            dut.data_in_1_valid, dut.data_in_1_ready
        )
        
        self.input_drivers = {
            "data_0": self.data_in_0_driver,
            "data_1": self.data_in_1_driver,
        }
            
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
        block_size = self.get_parameter("BLOCK_SIZE")
        return {
            "data_0": torch.randn((block_size)),
            "data_1": torch.randn((block_size)),
        }

    def generate_exp_outputs(self):
        block_size = self.get_parameter("BLOCK_SIZE")
        return torch.randn((block_size))

    async def run_test(self, us, num=1):
        await self.reset()
        self.data_out_0_monitor.ready.value = 1

        for i in range(num):
            # Generate random inputs
            inputs = self.generate_inputs()
            # Generate random expected outputs
            exp_out = self.generate_exp_outputs()
            
            # Process input data 0
            data_0_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["data_0"],
                config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1")
                },
                parallelism=[self.get_parameter("BLOCK_SIZE")]
            )
            self.data_in_0_driver.load_driver(data_0_inputs)

            # Process input data 1
            data_1_inputs = self.preprocess_tensor_for_mxint(
                tensor=inputs["data_1"],
                config={
                    "width": self.get_parameter("DATA_IN_1_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_1_PRECISION_1")
                },
                parallelism=[self.get_parameter("BLOCK_SIZE")]
            )
            self.data_in_1_driver.load_driver(data_1_inputs)

            # Load output monitor
            outs = self.preprocess_tensor_for_mxint(
                tensor=exp_out,
                config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
                parallelism=[self.get_parameter("BLOCK_SIZE")]
            )
            self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test_addition(dut):
    tb = AdditionTB(dut)
    await tb.run_test(us=10, num=5)

def get_addition_config(kwargs={}):
    """
    Default configuration for addition test
    """
    config = {
        # Input 0 precision
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 4,
        
        # Input 1 precision (same as input 0)
        "DATA_IN_1_PRECISION_0": 8,
        "DATA_IN_1_PRECISION_1": 4,
        
        # Output precision
        "DATA_OUT_0_PRECISION_0": 9,  # One extra bit for addition
        "DATA_OUT_0_PRECISION_1": 4,
        
        # Tensor dimensions
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 20,
        "DATA_IN_0_TENSOR_SIZE_DIM_2": 1,
        
        # Parallelism configuration
        "DATA_IN_0_PARALLELISM_DIM_0": 20,
        "DATA_IN_0_PARALLELISM_DIM_1": 20,
        "DATA_IN_0_PARALLELISM_DIM_2": 1,
    }

    config.update(kwargs)
    return config

def test_addition_regression():
    """
    Run regression tests with different configurations
    """
    mase_runner(
        trace=True,
        module_param_list=[
            # Basic test with default config
            get_addition_config(),
        ]
    )

if __name__ == "__main__":
    test_addition_regression()