#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import sys
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

from chop.passes.graph.transforms.quantize.quantized_modules import BatchNorm1dInteger

import torch
from queue import Queue

logger = logging.getLogger("tb_signals")
logger.setLevel(logging.DEBUG)


class BatchNormTB(Testbench):
    def __init__(self, dut, num_features=16) -> None: #, in_features=4, out_features=4
        super().__init__(dut, dut.clk, dut.rst) #needed to add rst signal for inheritance

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = StreamDriver(
            dut.clk, 
            dut.data_in_0, 
            dut.data_in_0_valid,
            dut.data_in_0_ready
        )

        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )

        self.bias_driver = StreamDriver(
            dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        )

        self.mean_driver = StreamDriver(
            dut.clk, dut.mean, dut.mean_valid, dut.mean_ready
        )
        
        self.num_features = num_features

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        
    
    def fe_model(self, data_in): 
        #TODO: implement a functionally equivalent model here 
        #TODO: combine with random testing
        return data_in

    def preprocess_tensor(self, tensor, quantizer, frac_width, parallelism):
        # print("BEFORE: ", tensor)
        tensor = quantizer(tensor)
        tensor = (tensor * 2 ** frac_width).int()
        # logger.info(f"\nTensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        # logger.info(f"\nTensor after reshaping: {tensor}")
        return tensor

    
    def postprocess_tensor(self, tensor, config):
        tensor = [item * (1.0/2.0) ** config["frac_width"] for item in tensor]
        return tensor


    def get_test_case(self, config, parallelism, out_parallelism):
        model = BatchNorm1dInteger(
            num_features=self.num_features,
            config=config                
        )
        
        # Train first to generate a running mean/average
        training_inputs = torch.randn((10, model.num_features))
        model.train(True)
        model(training_inputs)

        # Now run inference to generate expected model outputs
        model.train(False)
        inputs = torch.randn((1, model.num_features))
        print("Inputs: ", inputs)
        print("Inputs (quantized): ", model.w_quantizer(inputs))
        exp_outputs_model = model(inputs)
        print("Expected outputs model: ", exp_outputs_model)

        inputs = self.preprocess_tensor(
            inputs, 
            model.x_quantizer, 
            config["data_in_frac_width"], 
            int(parallelism)
        )
        print("Pre-processed inputs: ", inputs)

        print("Running variance: ", model.running_var)
        stdv = torch.tensor([var ** (1.0/2.0) for var in model.running_var])
        print("Stdv: ", stdv)

        print("model weight: ", model.weight)
        weight = model.weight / stdv
        print("weight: ", weight)
        weight = self.preprocess_tensor(
            weight, 
            model.w_quantizer, 
            config["weight_frac_width"], 
            int(parallelism)
        )

        bias = self.preprocess_tensor(
            model.bias, 
            model.b_quantizer, 
            config["bias_frac_width"], 
            int(parallelism)
        )
        
        print("Running mean: ", model.running_mean)
        mean = self.preprocess_tensor(
            model.running_mean, 
            model.b_quantizer, 
            config["weight_frac_width"], 
            int(parallelism)
        )

        data_width = config["data_in_width"]
        data_frac_width = config["data_in_frac_width"]

        exp_outputs_model = model.x_quantizer(exp_outputs_model)
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
        print("Exp. outputs model: ", exp_outputs_model)

        exp_outputs_model= (exp_outputs_model* 2 ** data_frac_width).int()
        exp_outputs_model= exp_outputs_model.reshape(-1, int(out_parallelism)).tolist()
        
        print("Exp. outputs model pre-processed: ", exp_outputs_model)
        
        return inputs, weight, bias, mean, exp_outputs_model

    
    async def run_test(self): 
        await self.reset()
        print(f'================= DEBUG: in run_test ================= \n')

        # Setup config dict.
        config = {
            "data_in_width": int(self.dut.DATA_IN_0_PRECISION_0),
            "data_in_frac_width": int(self.dut.DATA_IN_0_PRECISION_1),
            "weight_width": int(self.dut.WEIGHT_PRECISION_0),
            "weight_frac_width": int(self.dut.WEIGHT_PRECISION_1),
            "bias_width": int(self.dut.BIAS_PRECISION_0),
            "bias_frac_width": int(self.dut.BIAS_PRECISION_1),
        }
        parallelism     = int(self.dut.DATA_IN_0_PARALLELISM_DIM_0) * int(self.dut.DATA_IN_0_PARALLELISM_DIM_1)    
        out_parallelism = int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0) * int(self.dut.DATA_OUT_0_PARALLELISM_DIM_1)    

        # Get two sets of test case data.
        inputs_1, weight_1, bias_1, mean_1, exp_outputs_1 = self.get_test_case(config, parallelism, out_parallelism)
        inputs_2, weight_2, bias_2, mean_2, exp_outputs_2 = self.get_test_case(config, parallelism, out_parallelism)

        self.data_out_0_monitor.load_monitor(exp_outputs_1)
        self.data_out_0_monitor.load_monitor(exp_outputs_2)
        
        # Indicate we are ready to receive.
        self.data_out_0_monitor.ready.value = 1

        self.weight_driver.load_driver(weight_1)
        self.weight_driver.load_driver(weight_2)

        await Timer(10, units="ns")
        self.bias_driver.load_driver(bias_1)
        await Timer(10, units="ns")
        self.mean_driver.load_driver(mean_1)
        
        # Apply backpressure
        self.data_out_0_monitor.ready.value = 0
        self.bias_driver.load_driver(bias_2)
        await Timer(10, units="ns")        
        
        self.data_out_0_monitor.ready.value = 1
        self.data_in_0_driver.load_driver(inputs_1)
        await Timer(10, units="ns")
        
        self.data_in_0_driver.load_driver(inputs_2)
        self.mean_driver.load_driver(mean_2)
        print(f'================= DEBUG: put values on input ports ================= \n')

        print(f'================= DEBUG: generated values with fe model ================= \n')

        print(f'================= DEBUG: loaded hw outptus ================= \n')

        await Timer(300, units="ns")

        print(f'================= DEBUG: in run_test waited 1ms ================= \n')

@cocotb.test()
async def simple_test(dut):
        print(f'================= DEBUG: in simple_test ================= \n')
        tb = BatchNormTB(dut)
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
                "DATA_OUT_0_PARALLELISM_DIM_0": 4,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,                  
             },
             # {
             #    "DATA_IN_0_PRECISION_0": 8,
             #    "DATA_IN_0_PRECISION_1": 3,
             #    "DATA_IN_0_PARALLELISM_DIM_0": 32,
             #    "DATA_IN_0_PARALLELISM_DIM_1": 1,                  
             # },            
             # {
             #    "DATA_IN_0_PRECISION_0": 16,
             #    "DATA_IN_0_PRECISION_1": 5,
             #    "DATA_IN_0_PARALLELISM_DIM_0": 16,
             #    "DATA_IN_0_PARALLELISM_DIM_1": 1,                  
             # },
             # {
             #    "DATA_IN_0_PRECISION_0": 8,
             #    "DATA_IN_0_PRECISION_1": 3,
             #    "DATA_IN_0_PARALLELISM_DIM_0": 16,
             #    "DATA_IN_0_PARALLELISM_DIM_1": 1,
             #    "WEIGHT_PRECISION_0": 16,
             #    "WEIGHT_PRECISION_1": 5,
             # },
             # {
             #    "DATA_IN_0_PRECISION_0": 8,
             #    "DATA_IN_0_PRECISION_1": 3,
             #    "DATA_IN_0_PARALLELISM_DIM_0": 16,
             #    "DATA_IN_0_PARALLELISM_DIM_1": 1,
             #    "WEIGHT_PRECISION_0": 16,
             #    "WEIGHT_PRECISION_1": 5,
             #    "BIAS_PRECISION_0": 12,
             #    "BIAS_PRECISION_1": 4,
             # },
              
        ]    
    )        
