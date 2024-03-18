#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
from functools import partial
import numpy as np

import sys
# TODO: Remove these.
sys.path.insert(0,'/home/sv720/mase_fork/mase_group7/machop')
sys.path.insert(0,'/home/jlsand/mase_group7/machop')

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *
from cocotb.binary import BinaryValue

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer
from chop.passes.graph.transforms.quantize.quantized_modules import LayerNormInteger

import torch
from queue import Queue



logger = logging.getLogger("tb_signals")
logger.setLevel(logging.DEBUG)


class LayerNormTB(Testbench):
    def __init__(self, dut, num_features=16) -> None:
        super().__init__(dut, dut.clk, dut.rst) #needed to add rst signal for inheritance

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = StreamDriver(
            dut.clk, 
            dut.data_in_0, 
            dut.data_in_0_valid,
            dut.data_in_0_ready
        )        

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        
        self.config = {
            "data_in_width": 8,
            "data_in_frac_width": 3,
            "weight_width": 8,
            "weight_frac_width": 3,
            "bias_width": 8,
            "bias_frac_width": 3,
        }

        self.model = LayerNormInteger(num_features, elementwise_affine=False, config=self.config);
      
    def preprocess_tensor(self, tensor, quantizer, frac_width, parallelism):
        # print("BEFORE: ", tensor)
        tensor = quantizer(tensor)
        tensor = (tensor * (2 ** frac_width)).int()
        # logger.info(f"\nTensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        # logger.info(f"\nTensor after reshaping: {tensor}")
        return tensor

    
    def postprocess_tensor(self, tensor, config):
        tensor = [item * (1.0/2.0) ** config["frac_width"] for item in tensor]
        return tensor
    
    async def run_test(self): 
        await self.reset()
        print(f'================= DEBUG: in run_test ================= \n')

        # Store fixed point data
        # TODO(jlsand): We assume that the fixed point format is the same for all
        # parameters. Fair assumption or faulty?

        data_width = self.config["data_in_width"]
        data_frac_width = self.config["data_in_frac_width"]
        parallelism = int(self.dut.DATA_IN_0_PARALLELISM_DIM_0) * int(self.dut.DATA_IN_0_PARALLELISM_DIM_1)    
        local_config = {"width": data_width, "frac_width": data_frac_width}

        self.model.reset_parameters()
        print(self.model.weight)
        print(self.model.bias)
        self.model.training = False
        inputs = torch.randn((1, self.model.normalized_shape[0])) + 0.5
        
        quantizer = partial(
            integer_quantizer, width=data_width, frac_width=data_frac_width
        )
        
        print("Inputs: ", inputs)
        quantized_inputs = quantizer(inputs)
        print("Inputs (quantized): ", quantized_inputs)

        exp_outputs_model = self.model(inputs)
        print("Expected outputs model: ", exp_outputs_model)
        print("Expected sum of quantized inputs: ", quantizer(sum(quantized_inputs[0])))
        print("Expected mean of quantized inputs: ", quantizer(quantizer(sum(quantized_inputs[0]))/len(inputs[0])))
        print("Expected var of quantized inputs: ", quantizer(quantized_inputs[0].var()))
        # print("Expected stdv of quantized inputs: ", quantizer(np.var(quantized_inputs[0]) ** 0.5))
        
        inputs = self.preprocess_tensor(
            inputs, 
            quantizer, 
            data_frac_width, 
            parallelism
        )
        print("Pre-processed inputs: ", inputs)

        weight = self.preprocess_tensor(
            self.model.weight, 
            self.model.w_quantizer, 
            data_frac_width, 
            int(parallelism)
        )

        bias = self.preprocess_tensor(
            self.model.bias, 
            self.model.b_quantizer, 
            data_frac_width, 
            int(parallelism)
        )

        self.data_out_0_monitor.ready.value = 1
        print(f'================= DEBUG: asserted ready_out ================= \n')
        
        self.data_in_0_driver.load_driver(inputs)
        self.dut.weight.value = weight[0]; 
        self.dut.bias.value = bias[0]; 
        print(f'================= DEBUG: put values on input ports ================= \n')

        exp_outputs_model = quantizer(exp_outputs_model)
        print("Exp. outputs model: ", exp_outputs_model)

        # # The output from the module is treated as positive integers
        # # convert the negative expected outputs to their positive
        # # equivalent when not treated as 2's complement.
        def convert(x):
            if x >= 0:
                return x
            else:
                new_x = x + (2 ** (data_width-data_frac_width))
                print(x, " ", new_x)
                return new_x
        exp_outputs_model.detach().apply_(convert)
        # exp_outputs_model = exp_outputs_model.floor() 
        print("Exp. outputs model: ", exp_outputs_model)

        exp_outputs_model= (exp_outputs_model* 2 ** data_frac_width).int()
        exp_outputs_model= exp_outputs_model.reshape(-1, int(parallelism)).tolist()
        print(f'================= DEBUG: generated values with fe model ================= \n')

        self.data_out_0_monitor.load_monitor(exp_outputs_model)
        print(f'================= DEBUG: loaded hw outptus ================= \n')

        await Timer(1000, units="us")

        print(f'================= DEBUG: in run_test waited 1ms ================= \n')

@cocotb.test()
async def simple_test(dut):
        print(f'================= DEBUG: in simple_test ================= \n')
        tb = LayerNormTB(dut)
        print(f'================= DEBUG: initialized tb ================= \n')

        await tb.run_test()
        print(f'================= DEBUG: ran test ================= \n')

if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
             {
                "DATA_IN_0_PRECISION_0": 8,
                "DATA_IN_0_PRECISION_1": 3,
                "DATA_IN_0_PARALLELISM_DIM_0": 16,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,                  
             },
              
        ]    
    )
