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

class MxIntLayerNorm1DTB(Testbench):
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

        # Output monitor
        self.out_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

        self.input_drivers = {
            "data_in": self.data_in_driver,
        }
        self.output_monitors = {"out": self.out_monitor}
        
        # Model parameters
        self.tensor_size_dim_0 = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.parallelism_dim_0 = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        (qtensor, mtensor, etensor) = mxint_hardware(tensor, config, parallelism)
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

    async def run_test(self):
        await self.reset()
        self.log.info("Reset finished")
        self.out_monitor.ready.value = 1

        input_data = torch.randn((1, self.tensor_size_dim_0))
        # Update config to match RTL parameter names
        input_config = {
            "width": self.get_parameter("DATA_IN_0_MAN_WIDTH"),
            "exponent_width": self.get_parameter("DATA_IN_0_EXP_WIDTH"),
            "round_bits": 4,
        }

        input_parallelism = [
            1,
            self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
        ]
        (qtensor, mtensor, etensor) = mxint_hardware(input_data, input_config, input_parallelism)
        shape = mtensor.shape
        mtensor = mtensor.reshape(-1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")).unsqueeze(0)
        mtensor = mtensor // 2**(etensor.max() - etensor).unsqueeze(-1)
        etensor = etensor.max().repeat(etensor.shape)
        input_data_processed = pack_tensor_to_mx_listed_chunk(mtensor, etensor, input_parallelism) 
        self.data_in_driver.load_driver(input_data_processed)

        from mxint_quant.layernorm import mxint_layer_norm
        qinput = mtensor * 2**(etensor.unsqueeze(-1) - input_config["width"] - 1)
        qinput = qinput.reshape(shape)
        layer_norm_config = {
            "name": "mxint_hardware",
            # data
            "data_in_width": self.get_parameter("DATA_IN_0_MAN_WIDTH"),
            "data_in_exponent_width": self.get_parameter("DATA_IN_0_EXP_WIDTH"),
            "data_in_parallelism": [1, self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")],
            "data_out_width": self.get_parameter("DATA_OUT_0_MAN_WIDTH"), 
            "data_out_exponent_width": self.get_parameter("DATA_OUT_0_EXP_WIDTH"), 
            "data_out_parallelism": [1, self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0")],
        }
        int_config = {
            "qx_lossy": True,
            "num_val_0_lossy": True,
            "num_val_1_lossy": True,
            "mean_lossy": True,
            "var_lossy": True,
            "isqrt_lossy": True,
            "data_in_width": layer_norm_config["data_in_width"],
            "data_in_frac_width": layer_norm_config["data_in_width"] - 1,
            "isqrt_in_width": self.get_parameter("ISQRT_IN_MAN_WIDTH"),
            "isqrt_in_exponent_width": 6,
            "isqrt_out_width": self.get_parameter("ISQRT_OUT_MAN_WIDTH"),
            "isqrt_out_frac_width": self.get_parameter("ISQRT_OUT_MAN_FRAC_WIDTH"),
            "isqrt_out_exponent_width": 6,
            "weight_width": 8,
            "weight_frac_width": 6,
            "bias_width": 8,
            "bias_frac_width": 6,
            "data_out_width": self.get_parameter("DATA_OUT_0_MAN_WIDTH"),
            "data_out_frac_width": self.get_parameter("DATA_OUT_0_MAN_FRAC_WIDTH"),
        }
        qout_data, mout_data, eout_data = mxint_layer_norm(qinput, (self.tensor_size_dim_0,), None, None, q_config=int_config)
        eout_data = eout_data.repeat(etensor.shape)

        # Simplified parallelism config since RTL only has one dimension
        out_parallelism = [
            1,
            self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
        ]
        out_processed = pack_tensor_to_mx_listed_chunk(mout_data, eout_data, out_parallelism)

        self.out_monitor.load_monitor(out_processed)

        await Timer(100, units="us")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError("Output monitor is not empty at end of test")

@cocotb.test()
async def test_mxint_layer_norm(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MxIntLayerNorm1DTB(dut)
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
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 10,  # Changed from 8 to match RTL
    "DATA_IN_0_PARALLELISM_DIM_0": 2,   # Changed from 2 to match RTL

    # Data width parameters
    "DATA_IN_0_MAN_WIDTH": 8,           # Added to match RTL
    "DATA_IN_0_MAN_FRAC_WIDTH": 7,      # Added to match RTL
    "DATA_IN_0_EXP_WIDTH": 4,           # Added to match RTL
    
    "DATA_OUT_0_MAN_WIDTH": 8,          # Added to match RTL
    "DATA_OUT_0_MAN_FRAC_WIDTH": 7,     # Added to match RTL
    "DATA_OUT_0_EXP_WIDTH": 4,          # Added to match RTL

    # ISQRT parameters
    "ISQRT_IN_MAN_WIDTH": 8,            # Added to match RTL
    "ISQRT_IN_MAN_FRAC_WIDTH": 7,       # Added to match RTL
    "ISQRT_OUT_MAN_WIDTH": 8,           # Added to match RTL
    "ISQRT_OUT_MAN_FRAC_WIDTH": 4,      # Added to match RTL
}

def test_layer_norm_smoke():
    valid_width = default_config["ISQRT_IN_MAN_WIDTH"] + 1
    valid_frac_width = default_config["ISQRT_IN_MAN_WIDTH"] - 1

    out_width = default_config["ISQRT_OUT_MAN_WIDTH"]
    out_frac_width = default_config["ISQRT_OUT_MAN_FRAC_WIDTH"]

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
