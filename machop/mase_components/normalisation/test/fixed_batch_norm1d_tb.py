#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import sys
sys.path.insert(0,'/home/sv720/mase_fork/mase_group7/machop')

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

#import torch
from queue import Queue

logger = logging.getLogger("tb_signals")
logger.setLevel(logging.DEBUG)


class BatchNormTB(Testbench):
    def __init__(self, dut) -> None: #, in_features=4, out_features=4
        super().__init__(dut, dut.clk, dut.rst) #needed to add rst signal for inheritance

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_driver = StreamDriver(
            dut.clk, 
            dut.data_in, 
            dut.valid_in,
            dut.ready_in
        )

        self.data_out_monitor = StreamMonitor(
            dut.clk,
            dut.data_out,
            dut.valid_out,
            dut.ready_out,
            check=True,
        )
        """ 
        self.data_in_width = 8
        self.data_in_frac_width = 4
        self.data_out_width = 8
        self.data_out_frac_width = 4
        self.parallelism = 16
        """
    
    def fe_model(self, data_in): 
        #TODO: implement a functionally equivalent model here 
        #TODO: combine with random testing
        return data_in
    async def run_test(self): 
        print(f'================= DEBUG: in run_test ================= \n')

        
        #await self.reset()
        #print(f'================= DEBUG: rest successfully ================= \n')

        #logger.info(f"Reset finished")
        self.data_out_monitor.ready.value = 1
        print(f'================= DEBUG: asserted ready_out ================= \n')

        #input = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

        inputs = [[1] * 16 for _ in range(8)]

        self.data_in_driver.load_driver(inputs) #this needs to be a tensor
        print(f'================= DEBUG: put values on input ports ================= \n')

        exp_out = [[2] * 16 for _ in range(8)]#self.fe_model(inputs) #TODO: implement FE model
        print(f'================= DEBUG: generated values with fe model ================= \n')


        self.data_out_monitor.load_monitor(exp_out)
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
                "IN_FRAC_WIDTH": 4,
                "OUT_WIDTH": 8,
                "OUT_FRAC_WIDTH": 4,
                "PARALLELISM": 16,
                  
             }
              
        ]    
    )



    #def get_dut_parameters(self): #TODO: discuss need for this function
        
