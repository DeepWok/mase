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
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class fixed_logsigmoid_tb(Testbench):
    def __init__(self, dut, dut_params, num_words) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.data_width = dut_params["DATA_IN_0_PRECISION_0"] 
        self.frac_width = dut_params["DATA_IN_0_PRECISION_1"]

        self.outputwidth = dut_params["DATA_OUT_0_PRECISION_0"] 
        self.outputfracw = dut_params["DATA_OUT_0_PRECISION_1"]

        self.num_words_per_input = num_words

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )  

        self.dquantizer = partial(
            integer_quantizer, width=self.data_width, frac_width=self.frac_width, is_signed = True
        )

        self.model = torch.nn.LogSigmoid()

    def exp(self, inputs):
        # Run the model with the provided inputs and return the outputs
        # cond = torch.logical_not(torch.logical_and(inputs <= self.thresh*2**self.frac_width, inputs >= -1 * self.thresh *2**self.frac_width))
        # out = torch.where(cond, inputs, torch.tensor(0))
        # unsignedout = torch.where(out < 0, torch.tensor(out % (2**self.data_width)), out)
        # logger.info(f'IN EXP - Inputs: \n{inputs}')
        # logger.info(f'IN EXP - FLOAT Inputs: \n{inputs.float()}')

        m = self.model(inputs.float())
        # logger.info(f'IN EXP - FLOAT OUTPUT: \n{m}')
        m = self.dquantizer(m)
        # logger.info(f'IN EXP - DQ OUTPUT: \n{m}')
        # mout = m.clamp(min=-1*2**(self.outputwidth-1), max = 2**(self.outputwidth-1)-1)
        m2 = (m * 2 ** self.frac_width).to(torch.int64)
        m2 = torch.where(m2 < 0, (m2.clone().detach() % (2**self.outputwidth)), m2)
        return m2
        

    def generate_inputs(self):
        realinp = torch.randn(self.num_words_per_input)
        # realinp = torch.Tensor([-0.4519])
        # logger.info(f"Real input: \n{realinp}")
        inputs = self.dquantizer(realinp)
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
        for i in range(1000):
            inputs, real_inp = self.generate_inputs()
            inputs = inputs.tolist()
            exp_out = self.exp(real_inp)
            exp_out = exp_out.tolist()
            logger.info("Inputs and expected generated")
            
            self.data_in_0_driver.load_driver([inputs])
            # self.data_in_0_driver.append(inputs)
            self.data_out_0_monitor.load_monitor([exp_out])

        await Timer(1000, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    data_width = dut_params["DATA_IN_0_PRECISION_0"] 
    frac_width = dut_params["DATA_IN_0_PRECISION_1"]
    nw_per_input = dut_params["DATA_IN_0_PARALLELISM_DIM_0"] * dut_params["DATA_IN_0_PARALLELISM_DIM_1"]
    generate_memory.generate_mem("logsigmoid", data_width, frac_width)
    
    tb = fixed_logsigmoid_tb(dut, dut_params, num_words=nw_per_input)
    await tb.run_test()
  
dut_params = {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 10,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_IN_0_PRECISION_0": 5,
                "DATA_IN_0_PRECISION_1": 1,

                "DATA_OUT_0_PRECISION_0": 5,
                "DATA_OUT_0_PRECISION_1": 1,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 10,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,

            }

torch.manual_seed(1)
if __name__ == "__main__":
    mase_runner(
        module_param_list=[
            dut_params
        ]
    )
