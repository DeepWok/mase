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
from mxint_quant import mxint_quant_block, mxint_hardware
from utils import pack_tensor_to_mx_listed_chunk

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
            check=True
        )

        self.input_drivers = {
            "data_in": self.data_in_driver,
            "weight": self.weight_driver,
            "bias": self.bias_driver,
        }
        self.output_monitors = {"out": self.out_monitor}
        self.out_monitor.log.setLevel(logging.DEBUG)
        
        # Model parameters
        self.tensor_size_dim_0 = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.tensor_size_dim_1 = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1")
        self.parallelism_dim_0 = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")
        self.parallelism_dim_1 = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        (qtensor, mtensor, etensor) = mxint_hardware(tensor, config, parallelism)
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

        # Input data processing
        input_config = {
            "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            "round_bits": 4,
        }
        input_parallelism = [
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
        ]
        (qinput, minput, einput) = mxint_hardware(input_data, input_config, input_parallelism)
        # Save original shape
        shape = minput.shape
        
        # Reshape to match parallelism structure like in pack_tensor_to_mx_listed_chunk
        reshaped_mtensor = minput.reshape(-1, shape[-2] // self.parallelism_dim_1, 
                                        self.parallelism_dim_1,
                                        shape[-1] // self.parallelism_dim_0, 
                                        self.parallelism_dim_0)\
                                .permute(0, 1, 3, 2, 4)\
                                .reshape(-1, self.parallelism_dim_1 * self.parallelism_dim_0)
        
        # Get max exponent per block and adjust mantissa
        reshaped_etensor = einput.reshape(-1)
        emax = reshaped_etensor.max()
        reshaped_mtensor = reshaped_mtensor // 2**(emax - reshaped_etensor).unsqueeze(-1)
        
        # Reshape back to original shape
        minput = reshaped_mtensor.reshape(-1, shape[-2] // self.parallelism_dim_1,
                                        shape[-1] // self.parallelism_dim_0,
                                        self.parallelism_dim_1,
                                        self.parallelism_dim_0)\
                                .permute(0, 1, 3, 2, 4)\
                                .reshape(shape)
        einput = einput.max().repeat(einput.shape)
        input_data_processed = pack_tensor_to_mx_listed_chunk(minput, einput, input_parallelism)

        # Weight processing  
        weight_config = {
            "width": self.get_parameter("WEIGHT_PRECISION_0"),
            "exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
            "round_bits": 4,
        }
        # Weight has shape (tensor_size_dim_0,)
        weight_parallelism = [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")]
        weight_processed = self.preprocess_tensor_for_mxint(weight, weight_config, weight_parallelism)

        # Bias processing
        bias_config = {
            "width": self.get_parameter("BIAS_PRECISION_0"),
            "exponent_width": self.get_parameter("BIAS_PRECISION_1"),
            "round_bits": 4,
        }
        # Bias has shape (tensor_size_dim_0,)
        bias_parallelism = [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")]
        bias_processed = self.preprocess_tensor_for_mxint(bias, bias_config, bias_parallelism)

        # Load drivers
        self.data_in_driver.load_driver(input_data_processed)
        self.weight_driver.load_driver(weight_processed)
        self.bias_driver.load_driver(bias_processed)

        # Generate expected output
        from mxint_quant.layernorm import mxint_layer_norm
        qinput = minput * 2**(einput.reshape(-1)[0] - input_config["width"] - 1)
        int_config = {
            "qx_lossy": True,
            "num_val_0_lossy": True,
            "num_val_1_lossy": True,
            "mean_lossy": True,
            "var_lossy": True,
            "isqrt_lossy": True,
            "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_0") - 1,
            "isqrt_in_width": self.get_parameter("ISQRT_IN_PRECISION_0"),
            "isqrt_in_exponent_width": 6,
            "isqrt_out_width": self.get_parameter("ISQRT_OUT_PRECISION_0"),
            "isqrt_out_frac_width": self.get_parameter("ISQRT_OUT_PRECISION_1"),
            "isqrt_out_exponent_width": 6,
            "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
            "weight_exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
            "weight_parallelism": [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")],
            "bias_width": self.get_parameter("BIAS_PRECISION_0"), 
            "bias_exponent_width": self.get_parameter("BIAS_PRECISION_1"),
            "bias_parallelism": [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")],
            "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "data_out_exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            "data_out_parallelism": [self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"), self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0")],
        }
        qout_data, mout_data, eout_data = mxint_layer_norm(qinput, (self.tensor_size_dim_0,), weight, bias, q_config=int_config)
        eout_data = eout_data

        out_parallelism = [
            self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
            self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
        ]
        out_processed = pack_tensor_to_mx_listed_chunk(mout_data, eout_data, out_parallelism)
        self.out_monitor.load_monitor(out_processed)

        await Timer(100, units="us")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError("Output monitor is not empty at end of test")

@cocotb.test()
async def test_mxint_layer_norm(dut):
    # cocotb.start_soon(check_signal(dut))
    tb = MxIntLayerNormTB(dut)
    await tb.run_test()

async def check_signal(dut):
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        print(dut.data_in_0_valid.value, dut.data_in_0_ready.value)
        print("end")

default_config = {
    # Input/output dimensions
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
    "DATA_IN_0_TENSOR_SIZE_DIM_2": 1,
    "DATA_IN_0_PARALLELISM_DIM_0": 2,
    "DATA_IN_0_PARALLELISM_DIM_1": 1,
    "DATA_IN_0_PARALLELISM_DIM_2": 1,

    # Data width parameters
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    
    "WEIGHT_PRECISION_0": 8,
    "WEIGHT_PRECISION_1": 4,
    
    "BIAS_PRECISION_0": 8,
    "BIAS_PRECISION_1": 4,
    
    "DATA_OUT_0_PRECISION_0": 8,
    "DATA_OUT_0_PRECISION_1": 4,

    # ISQRT parameters
    "ISQRT_IN_PRECISION_0": 8,
    "ISQRT_IN_PRECISION_1": 8,
    "ISQRT_OUT_PRECISION_0": 8,
    "ISQRT_OUT_PRECISION_1": 4,
    
    # Norm parameters
    "NORM_OUT_PRECISION_0": 8,
    "NORM_OUT_PRECISION_1": 4,
    
    # Other parameters
    "ELEMENTWISE_AFFINE": 1,
    "HAS_BIAS": 1
}

def test_layer_norm_smoke():
    valid_width = default_config["ISQRT_IN_PRECISION_0"] + 1
    valid_frac_width = default_config["ISQRT_IN_PRECISION_0"] - 1

    out_width = default_config["ISQRT_OUT_PRECISION_0"]
    out_frac_width = default_config["ISQRT_OUT_PRECISION_1"]

    from mase_components.helper import generate_memory
    generate_memory.generate_sv_lut(
        "isqrt",
        valid_width,
        valid_frac_width,
        out_width,
        out_frac_width,
        path=Path(__file__).parents[1] / "rtl",
        constant_mult=1,
        floor=False,
    )
    mase_runner(
        trace=True,
        module_param_list=[default_config],
        skip_build=False,
        sim="verilator",
    )

if __name__ == "__main__":
    test_layer_norm_smoke()
