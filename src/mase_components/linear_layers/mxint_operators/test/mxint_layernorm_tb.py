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

class MxIntLayerNormTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # Input data driver
        self.data_in_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready
        )
        
        # Weight driver
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mweight, dut.eweight),
            dut.weight_valid,
            dut.weight_ready
        )
        
        # Bias driver
        self.bias_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mbias, dut.ebias),
            dut.bias_valid,
            dut.bias_ready
        )

        # Output monitor
        self.out_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False
        )

        self.input_drivers = {
            "data_in": self.data_in_driver,
            "weight": self.weight_driver,
            "bias": self.bias_driver,
        }
        self.output_monitors = {"out": self.out_monitor}
        
        # Model parameters
        self.tensor_size_dim_0 = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.tensor_size_dim_1 = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1")
        self.parallelism_dim_0 = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")
        self.parallelism_dim_1 = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")

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

        # Generate random tensors for testing
        input_data = torch.randn((self.tensor_size_dim_1, self.tensor_size_dim_0))
        weight = torch.randn((self.tensor_size_dim_0,))
        bias = torch.randn((self.tensor_size_dim_0,))

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

        # Parallelism configurations
        input_parallelism = [
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
        ]
        weight_parallelism = [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")]
        bias_parallelism = [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")]

        # Preprocess all inputs
        input_data_processed = self.preprocess_tensor_for_mxint(input_data, input_config, input_parallelism)
        weight_processed = self.preprocess_tensor_for_mxint(weight, weight_config, weight_parallelism)
        bias_processed = self.preprocess_tensor_for_mxint(bias, bias_config, bias_parallelism)

        # Load all drivers
        self.data_in_driver.load_driver(input_data_processed)
        self.weight_driver.load_driver(weight_processed)
        self.bias_driver.load_driver(bias_processed)

        # Generate expected output (for verification)
        exp_out = torch.layer_norm(input_data, (self.tensor_size_dim_0,), weight, bias)
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

        await Timer(100, units="us")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError("Output monitor is not empty at end of test")

@cocotb.test()
async def test_mxint_layer_norm(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MxIntLayerNormTB(dut)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        print(dut.data_in_0_valid.value, dut.data_in_0_ready.value)
        # print(dut.parallel_dim_1[0].layer_norm_inst.edata_out_0.value, dut.parallel_norm_out_ready.value)
        # breakpoint()
        # if dut.data_in_0_valid.value == 1 and dut.data_in_0_ready.value == 1:
        #     print(
        #         "data_in_0 = ", [x.signed_integer for x in dut.mdata_in_0.value]
        #     )
        #     print(
        #         "shift_result = ", [x.signed_integer for x in dut.shift_result.value]
        #     )
        #     print(
        #         "clamped_n = ", [x.signed_integer for x in dut.clamped_n.value]
        #     )
        print("end")
default_config = {
    # Input/output dimensions
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 8,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
    "DATA_IN_0_PARALLELISM_DIM_0": 2,
    "DATA_IN_0_PARALLELISM_DIM_1": 2,

    # Precision parameters
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "WEIGHT_PRECISION_0": 8,
    "WEIGHT_PRECISION_1": 4,
    "BIAS_PRECISION_0": 8,
    "BIAS_PRECISION_1": 4,
    
    # ISQRT parameters
    "ISQRT_IN_PRECISION_0": 8,
    "ISQRT_IN_PRECISION_1": 7,
    "ISQRT_OUT_PRECISION_0": 8,
    "ISQRT_OUT_PRECISION_1": 4,
    
    # Layer norm specific parameters
    "ELEMENTWISE_AFFINE": 1,
    "NORM_OUT_PRECISION_0": 8,
    "NORM_OUT_PRECISION_1": 4,
    
    # Output parameters
    "DATA_OUT_0_PRECISION_0": 8,
    "DATA_OUT_0_PRECISION_1": 4,
}

def test_layer_norm_smoke():
    mase_runner(
        trace=True,
        module_param_list=[default_config],
        skip_build=False,
        sim="verilator",
        
    )

if __name__ == "__main__":
    test_layer_norm_smoke()
