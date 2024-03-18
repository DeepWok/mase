#!/usr/bin/env python3

import os, logging
import generate_memory
import pdb
from bitstring import BitArray
import cocotb
from functools import partial
from cocotb.triggers import *
from chop.passes.graph.transforms.quantize.quantizers import *
from mase_cocotb.testbench import Testbench 
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor, StreamMonitorFloat
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t
from math import ceil
# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

def split_and_flatten_2d_tensor(input_tensor, row_block_size, col_block_size):
    rows, cols = input_tensor.size()
    
    num_row_blocks = rows // row_block_size
    num_col_blocks = cols // col_block_size
    
    reshaped_tensor = input_tensor.view(num_row_blocks, row_block_size, num_col_blocks, col_block_size)
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
    flattened_tensor = reshaped_tensor.view(-1, row_block_size * col_block_size)
    return flattened_tensor

class fixed_softmax_tb(Testbench):
    def __init__(self, dut, dut_params) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.data_width = dut_params["DATA_IN_0_PRECISION_0"] 
        self.frac_width = dut_params["DATA_IN_0_PRECISION_1"]

        self.outputwidth = dut_params["DATA_OUT_0_PRECISION_0"] 
        self.outputfracw = dut_params["DATA_OUT_0_PRECISION_1"]
        
        self.num_in_features = dut_params["DATA_IN_0_TENSOR_SIZE_DIM_0"]
        self.num_in_batches = dut_params["DATA_IN_0_TENSOR_SIZE_DIM_1"]
        
        self.size_in_feature_blocks = dut_params["DATA_IN_0_PARALLELISM_DIM_0"]
        self.size_in_batch_blocks = dut_params["DATA_IN_0_PARALLELISM_DIM_1"]
        
        self.num_in_feature_splits = int(ceil(self.num_in_features / self.size_in_feature_blocks))
        self.num_in_batch_splits = int(ceil(self.num_in_batches / self.size_in_batch_blocks))
        
        self.num_out_features = dut_params["DATA_OUT_0_TENSOR_SIZE_DIM_0"]
        self.num_out_batches = dut_params["DATA_OUT_0_TENSOR_SIZE_DIM_1"]
        
        self.size_out_feature_blocks = dut_params["DATA_OUT_0_PARALLELISM_DIM_0"]
        self.size_out_batch_blocks = dut_params["DATA_OUT_0_PARALLELISM_DIM_1"]
        
        self.num_out_feature_splits = int(ceil(self.num_out_features / self.size_out_feature_blocks))
        self.num_out_batch_splits = int(ceil(self.num_out_batches / self.size_out_batch_blocks))
        
        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_0_monitor = StreamMonitorFloat(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready, self.outputwidth, self.outputfracw
        )  

        self.in_dquantizer = partial(
            integer_quantizer, width=self.data_width, frac_width=self.frac_width, is_signed = True
        )
        
        self.out_dquantizer = partial(
            integer_quantizer, width=self.outputwidth, frac_width=self.outputfracw, is_signed = True
        )

        self.model = torch.nn.Softmax()
        self.real_tensor = torch.randn(self.num_in_batches, self.num_in_features)
        logger.info(f"REAL TENSOR: \n{self.real_tensor}")
        
    def exp(self):
        # Run the model with the provided inputs and return the outputs
        # cond = torch.logical_not(torch.logical_and(inputs <= self.thresh*2**self.frac_width, inputs >= -1 * self.thresh *2**self.frac_width))
        # out = torch.where(cond, inputs, torch.tensor(0))
        # unsignedout = torch.where(out < 0, torch.tensor(out % (2**self.data_width)), out)
        # logger.info(f'IN EXP - Inputs: \n{inputs}')
        # logger.info(f'IN EXP - FLOAT Inputs: \n{inputs.float()}')
        # inputs = torch.flip(inputs, [1]) # match endianess of verilog
        
        m = self.model(self.real_tensor)
        
        
        # I think this is needed for endian matching when IN_PLL != OUT_PLL (maybe not w s-window?)
        
        
        # m = split_and_flatten_2d_tensor(m, self.size_in_batch_blocks, self.size_in_feature_blocks) # match input
        # m = torch.flip(m, [1]) # match endianess of verilog
         
        # m = torch.reshape(m, (self.num_out_batches, self.num_out_features)) # set to output tensor shape
        m = split_and_flatten_2d_tensor(m, self.size_out_batch_blocks, self.size_out_feature_blocks) # match output
        
        
        logger.info(f'EXP - FLOAT OUTPUT: \n{m}')
        m = self.out_dquantizer(m)
        # mout = m.clamp(min=-1*2**(self.outputwidth-1), max = 2**(self.outputwidth-1)-1)
        print(f"Output width: {self.outputwidth}, output frac width: {self.outputfracw}")
        
        m2 = (m * 2 ** self.outputfracw).to(torch.int64)
        print(m2)
        m2 = torch.where(m2 < 0, (m2.clone().detach() % (2**self.outputwidth)), m2)
        
        return m2
    
    def generate_inputs(self):
        
        
        # realinp = torch.Tensor([-0.4519])
        # logger.info(f"Real input: \n{realinp}")
        inputs = split_and_flatten_2d_tensor(self.real_tensor, self.size_in_batch_blocks, self.size_in_feature_blocks)
        logger.info(f"EXP - FLOAT INPUT: \n{inputs}")
        inputs = self.in_dquantizer(inputs)
        # inputs = inputs.reshape(self.num_in_feature_blocks, self.num_in_batch_blocks, self.num_in_feature_splits, self.num_in_batch_splits)
        
        # logger.info(f"Input: \n{inputs}")
        intinp = (inputs * 2**self.frac_width).to(torch.int64)
        return intinp, inputs

    def doubletofx(self, num, data_width, f_width, type = "bin"):
        assert type == "bin" or type == "hex", "type can only be: 'hex' or 'bin'"
        intnum = int(num * 2**(f_width))
        intbits = BitArray(int=intnum, length=data_width)
        return str(intbits.bin) if type == 'bin' else str(intbits)

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for i in range(1):
            inputs, real_tensor = self.generate_inputs()
            exp_out = self.exp()
            logger.info(f"real inputs: {real_tensor}")
            inputs = inputs.tolist()
            exp_out = exp_out.tolist()
            logger.info("Inputs and expected generated")
            logger.info(f"DUT IN: {inputs}")
            logger.info(f"DUT EXP OUT: {exp_out}")
            self.data_in_0_driver.load_driver(inputs)
            # self.data_in_0_driver.append(inputs)
            self.data_out_0_monitor.load_monitor(exp_out)

        await Timer(1000, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    in_data_width = dut_params["DATA_IN_0_PRECISION_0"] 
    in_frac_width = dut_params["DATA_IN_0_PRECISION_1"]
    out_data_width = dut_params["DATA_OUT_0_PRECISION_0"] 
    out_frac_width = dut_params["DATA_OUT_0_PRECISION_1"]
    generate_memory.generate_mem("exp", in_data_width, in_frac_width, out_data_width, out_frac_width)
    print("Generated memory")
    tb = fixed_softmax_tb(dut, dut_params)
    await tb.run_test()
  
dut_params = {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 12,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 12,
                
                "DATA_IN_0_PARALLELISM_DIM_0": 4,
                "DATA_IN_0_PARALLELISM_DIM_1": 3,
                
                "DATA_IN_0_PRECISION_0": 16,
                "DATA_IN_0_PRECISION_1": 8,

                "DATA_OUT_0_PRECISION_0": 16,
                "DATA_OUT_0_PRECISION_1": 8,
                
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 12,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 12,
                
                "DATA_OUT_0_PARALLELISM_DIM_0": 4,
                "DATA_OUT_0_PARALLELISM_DIM_1": 3,
            }

torch.manual_seed(1)
if __name__ == "__main__":
    # generate_memory.generate_mem("exp", dut_params["DATA_IN_0_PRECISION_0"], dut_params["DATA_IN_0_PRECISION_1"])
    mase_runner(
        module_param_list=[
            dut_params
        ]
    )
