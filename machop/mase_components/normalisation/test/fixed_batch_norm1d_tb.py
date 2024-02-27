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

        # self.weight_driver = StreamDriver(
        #     dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        # )

        # self.bias_driver = StreamDriver(
        #     dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        # )
        

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        """ 
        self.data_in_width = 8
        self.data_in_frac_width = 4
        self.data_out_width = 8
        self.data_out_frac_width = 4
        self.parallelism = 16
        """

        self.model = BatchNorm1dInteger(
            num_features,                
            config={
                "data_in_width": 8,
                "data_in_frac_width": 3,
                "weight_width": 8,
                "weight_frac_width": 3,
                "bias_width": 8,
                "bias_frac_width": 3,
            },
        )
    
    def fe_model(self, data_in): 
        #TODO: implement a functionally equivalent model here 
        #TODO: combine with random testing
        return data_in

    def preprocess_tensor(self, tensor, quantizer, config, parallelism):
        print("BEFORE: ", tensor)
        tensor = quantizer(tensor)
        tensor = (tensor * 2 ** config["frac_width"]).int()
        logger.info(f"\nTensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        logger.info(f"\nTensor after reshaping: {tensor}")
        return tensor
    
    async def run_test(self): 
        await self.reset()
        print(f'================= DEBUG: in run_test ================= \n')


        # Train first to generate a running mean/average
        training_inputs = torch.randn((10, self.model.num_features))
        self.model(training_inputs)

        self.model.training = False

        inputs = torch.randn((1, self.model.num_features))
        exp_outputs = self.model(inputs)
        # print("EXP OUTPUTS", exp_outputs)
        
        
        inputs = self.preprocess_tensor(
            inputs, 
            self.model.x_quantizer, 
            {"width": 8, "frac_width": 3}, 
            int(self.dut.PARALLELISM)
        )

        print("INPUT SIZE:", len(inputs), len(inputs[0]))

        gamma = self.preprocess_tensor(
            self.model.weight, 
            self.model.w_quantizer, 
            {"width": 8, "frac_width": 3}, 
            int(self.dut.PARALLELISM)
        )

        # print(self.model.bias)
        # beta = self.preprocess_tensor(
        #     self.model.bias, 
        #     self.model.b_quantizer, 
        #     {"width": 8, "frac_width": 3}, 
        #     1
        # )
        # print(beta)

        # stdv = torch.tensor([var ** (1.0/2.0) for var in self.model.running_var])
        # stdv = self.preprocess_tensor(
        #     stdv, 
        #     self.model.w_quantizer, 
        #     {"width": 8, "frac_width": 3}, 
        #     1
        # )

        # mean = self.preprocess_tensor(
        #     self.model.running_mean, 
        #     self.model.b_quantizer, 
        #     {"width": 8, "frac_width": 3}, 
        #     1
        # )


        self.data_out_0_monitor.ready.value = 1
        print(f'================= DEBUG: asserted ready_out ================= \n')

        print("MODEL VALUES: ", self.model.weight)
        
        print(self.dut.gamma.value)
        self.dut.gamma.value = gamma
        # print(self.dut.gamma.value)
        # print(len(self.dut.gamma.value)) # 16
        # print(len(self.dut.gamma.value[0])) # 8

        # self.dut.gamma.value = [BinaryValue(bytes(gamma[j])) for j in range(16)]
        # self.dut.gamma.value = [BinaryValue(0) * 16 for _ in range(16)]
        # self.dut.beta.value =  [BinaryValue(0) * 16 for _ in range(16)]
        # self.dut.gamma.value = [BinaryValue(1) * 16 for _ in range(16)]
        # self.dut.stdv.value =  [BinaryValue(1) * 16 for _ in range(16)]

        # inputs = [[1] * 8 for _ in range(16)]

        self.data_in_0_driver.load_driver(inputs) #this needs to be a tensor
        print(f'================= DEBUG: put values on input ports ================= \n')


        exp_outputs_quant = self.preprocess_tensor(
            exp_outputs, 
            self.model.x_quantizer, 
            {"width": 8, "frac_width": 3}, 
            int(self.dut.PARALLELISM)
        )

        # exp_out = [[1] * 8 for _ in range(16)]#self.fe_model(inputs) #TODO: implement FE model
        print(f'================= DEBUG: generated values with fe model ================= \n')


        self.data_out_0_monitor.load_monitor(exp_outputs_quant)
        print(f'================= DEBUG: loaded hw outptus ================= \n')
        

        await Timer(1000, units="us")
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
                "IN_WIDTH": 8,
                "IN_FRAC_WIDTH": 3,
                "IN_DEPTH": 16,
                "PARALLELISM": 16,                  
             }
              
        ]    
    )



    #def get_dut_parameters(self): #TODO: discuss need for this function
        
