#!/usr/bin/env python3

import pytest
import os, logging
from src.mase_components.helper import generate_memory
import pdb
from bitstring import BitArray
import cocotb
from functools import partial
from cocotb.triggers import *
from chop.nn.quantizers import integer_quantizer
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    StreamMonitorFloat,
)
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t
from math import ceil
from pathlib import Path

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


def split_and_flatten_2d_tensor(input_tensor, row_block_size, col_block_size):
    rows, cols = input_tensor.size()

    num_row_blocks = rows // row_block_size
    num_col_blocks = cols // col_block_size

    reshaped_tensor = input_tensor.view(
        num_row_blocks, row_block_size, num_col_blocks, col_block_size
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
    flattened_tensor = reshaped_tensor.view(-1, row_block_size * col_block_size)
    return flattened_tensor


class fixed_softmax_tb(Testbench):
    def __init__(self, module, dut, float_test=False) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)
        self.data_width = self.get_parameter("DATA_IN_0_PRECISION_0")
        self.frac_width = self.get_parameter("DATA_IN_0_PRECISION_1")
        
        self.exp_data_width = self.get_parameter("DATA_INTERMEDIATE_0_PRECISION_0")
        self.exp_frac_width = self.get_parameter("DATA_INTERMEDIATE_0_PRECISION_1")

        self.outputwidth = self.get_parameter("DATA_OUT_0_PRECISION_0")
        self.outputfracw = self.get_parameter("DATA_OUT_0_PRECISION_1")

        self.num_in_features = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.num_in_batches = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1")

        self.size_in_feature_blocks = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")
        self.size_in_batch_blocks = self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")

        self.num_in_feature_splits = int(
            ceil(self.num_in_features / self.size_in_feature_blocks)
        )
        self.num_in_batch_splits = int(
            ceil(self.num_in_batches / self.size_in_batch_blocks)
        )

        self.num_out_features = self.get_parameter("DATA_OUT_0_TENSOR_SIZE_DIM_0")
        self.num_out_batches = self.get_parameter("DATA_OUT_0_TENSOR_SIZE_DIM_1")

        self.size_out_feature_blocks = self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0")
        self.size_out_batch_blocks = self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1")

        self.num_out_feature_splits = int(
            ceil(self.num_out_features / self.size_out_feature_blocks)
        )
        self.num_out_batch_splits = int(
            ceil(self.num_out_batches / self.size_out_batch_blocks)
        )

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        if float_test:
            self.data_out_0_monitor = StreamMonitorFloat(
                dut.clk,
                dut.data_out_0,
                dut.data_out_0_valid,
                dut.data_out_0_ready,
                self.outputwidth,
                self.outputfracw,
            )
        else:
            self.data_out_0_monitor = StreamMonitor(
                dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
            )

        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)
        self.in_dquantizer = partial(
            integer_quantizer,
            width=self.data_width,
            frac_width=self.frac_width,
            is_signed=True,
        )

        self.exp_dquantizer = partial(
            integer_quantizer,
            width=self.exp_data_width,
            frac_width=self.exp_frac_width,
            is_signed=True,
        )
        
        self.out_dquantizer = partial(
            integer_quantizer,
            width=self.outputwidth,
            frac_width=self.outputfracw,
            is_signed=True,
        )

        self.model = module


    def exp(self):
        # Run the model with the provided inputs and return the expected integer outputs in the format expected by the monitor
        m = split_and_flatten_2d_tensor(
            self.real_out_tensor,
            self.size_out_batch_blocks,
            self.size_out_feature_blocks,
        )  # match output
        logger.debug(f"EXP - FLOAT OUTPUT: \n{m}")
        m = self.out_dquantizer(m)
        logger.debug(f"m_after_dquantizer: \n{m}")
        m2 = (m * 2**self.outputfracw).to(torch.int64)
        m2 = m2.clone().detach() % (2**self.outputwidth)

        return m2

    def generate_inputs(self):
        # Generate the integer inputs for the DUT in the format expected by the driver
        
        self.real_in_tensor = torch.randn(self.num_in_batches, self.num_in_features)
        self.quant_in_tensor = self.in_dquantizer(self.real_in_tensor)
        self.exp_tensor = self.exp_dquantizer(self.quant_in_tensor.exp())
        self.exp_sum = self.exp_tensor.sum(dim=-1)
        self.extended_quotient_tensor = ((self.exp_tensor*(2**self.exp_frac_width)) // self.exp_sum.expand(self.num_in_features, self.num_in_batches).permute(1,0))
        logger.info(self.extended_quotient_tensor)
        self.real_out_tensor = self.extended_quotient_tensor/(2**self.exp_frac_width)
        self.real_out_tensor_hw = self.real_out_tensor*(2**self.get_parameter("DATA_INTERMEDIATE_0_PRECISION_1"))
        logger.info(f"REAL IN TENSOR: \n{self.real_in_tensor}")
        logger.info(f"REAL OUT TENSOR: \n{self.real_out_tensor}")
        logger.info(f"REAL OUT TENSOR HW: \n{self.real_out_tensor_hw}")
        inputs = split_and_flatten_2d_tensor(
            self.real_in_tensor, self.size_in_batch_blocks, self.size_in_feature_blocks
        )
        logger.info(f"FLOAT INPUT: \n{inputs}")
        inputs = self.in_dquantizer(inputs)
        intinp = (inputs * 2**self.frac_width).to(torch.int64)
        return intinp, inputs

    def doubletofx(self, num, data_width, f_width, type="bin"):
        assert type == "bin" or type == "hex", "type can only be: 'hex' or 'bin'"
        intnum = int(num * 2 ** (f_width))
        intbits = BitArray(int=intnum, length=data_width)
        return str(intbits.bin) if type == "bin" else str(intbits)

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for i in range(10):
            inputs, real_tensor = self.generate_inputs()
            exp_out = self.exp()
            inputs = inputs.tolist()
            exp_out = exp_out.tolist()
            logger.info("Inputs and expected generated")
            logger.info(f"DUT IN: {inputs}")
            logger.info(f"DUT EXP OUT: {exp_out}")
            self.data_in_0_driver.load_driver(inputs)
            self.data_out_0_monitor.load_monitor(exp_out)

        await Timer(10, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = fixed_softmax_tb(torch.nn.Softmax(), dut, float_test=False)
    await tb.run_test()

dut_params = {
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 12,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
    "DATA_IN_0_PARALLELISM_DIM_0": 3,
    "DATA_IN_0_PARALLELISM_DIM_1": 2,
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 6,
    "DATA_INTERMEDIATE_0_PRECISION_0": 12,
    "DATA_INTERMEDIATE_0_PRECISION_1": 8,
    "DATA_OUT_0_PRECISION_0": 8,
    "DATA_OUT_0_PRECISION_1": 7,
}

torch.manual_seed(1)


@pytest.mark.dev
def test_fixed_softmax():
    path = Path(__file__).parents[1] / "rtl"
    generate_memory.generate_sv_lut(
        "exp",
        dut_params["DATA_IN_0_PRECISION_0"],
        dut_params["DATA_IN_0_PRECISION_1"],
        dut_params["DATA_INTERMEDIATE_0_PRECISION_0"],
        dut_params["DATA_INTERMEDIATE_0_PRECISION_1"],
        path=path
    )
    print("Generated memory")
    mase_runner(
        trace=True,module_param_list=[dut_params])


if __name__ == "__main__":
    test_fixed_softmax()
