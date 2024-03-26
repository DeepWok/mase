#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *
from mase_cocotb.random_test import RandomSource, RandomSink, check_results

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer
import torch.nn.init as init

import torch
print(torch.__version__)

logger = logging.getLogger("testbench")
# logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)
global_tensor_size = 6



# LLMint_monitor()



class LinearTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        
        self.in_features = dut.TENSOR_SIZE_DIM.value
        self.out_features = dut.TENSOR_SIZE_DIM.value
        self.high_slots = dut.HIGH_SLOTS.value
        self.threshold = dut.THRESHOLD.value
        self.bitwidth = dut.ORIGINAL_PRECISION.value
        self.reduced_bitwidth = dut.REDUCED_PRECISION.value
        self.weights_size = [dut.WEIGHT_DIM_0.value, dut.WEIGHT_DIM_1.value]
        self.dut = dut

       
        #----------------Drivers---------------
        #print(self.in_features)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_driver = StreamDriver(
            dut.clk, dut.data_in, dut.data_in_valid, dut.data_in_ready
        )

        self.weight_driver = StreamDriver(
            dut.clk, dut.weights, dut.weight_valid, dut.weight_ready
        )

        self.q_weight_driver = StreamDriver(
            dut.clk, dut.q_weights, dut.q_weight_valid, dut.q_weight_ready
        )

        # For latter if we tackle bias
        '''
        if int(dut.HAS_BIAS) == 1:
            self.bias_driver = StreamDriver(
                dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
            )
        '''
        #----------------Monitor B---------------

        self.data_out_monitor = StreamMonitor(
            dut.clk,
            dut.data_out,
            dut.data_out_valid,
            dut.data_out_ready,
            check=True,
        )

        #----------------Linear low---------------

        self.linear_low = LinearInteger(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False,
            config={
                "data_in_width": self.reduced_bitwidth,
                "data_in_frac_width": 0,
                "weight_width": self.reduced_bitwidth,
                "weight_frac_width": 0,
                "bias_width": self.reduced_bitwidth,
                "bias_frac_width": 0,
            },
        )

        #----------------Linear high--------------

        self.linear_high= LinearInteger(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False,
            config={
                "data_in_width": self.bitwidth,
                "data_in_frac_width": 0,
                "weight_width": self.bitwidth,
                "weight_frac_width": 0,
                "bias_width": self.bitwidth,
                "bias_frac_width": 0,
            },
        )

        self.initialize_weights(self.linear_high, mean=0.0, std=3.0)  # Adjust mean and std as needed for larger values
        self.linear_low.weight = self.linear_high.weight # Both linear layers should have the same weights


    def gather(self, x_low, x_high):
        return x_low + x_high
    
    def scatter(self, x):
        high_mat = []
        low_mat = []
        count_high = 0

        for k in reversed(x[0].tolist()):
            if abs(k) > self.threshold and count_high < self.high_slots:
                high_mat.append(k)
                low_mat.append(0)
                count_high += 1
            else:
                high_mat.append(0)
                low_mat.append(k)

        low_mat.reverse()
        high_mat.reverse()

        print('low_mat',low_mat)
        print('high_mat',high_mat)

        return torch.tensor(low_mat), torch.tensor(high_mat)
    

    def LLMint_model(self, inputs):
        x_low_i, x_high_i = self.scatter(inputs)

        x_low_i = self.linear_low.x_quantizer(x_low_i).int()

        x_low_o = self.linear_low(x_low_i)
        x_high_o = self.linear_high(x_high_i)

        outputs = self.gather(x_low_o, x_high_o)

        return outputs
    

    def initialize_weights(self, layer, mean=0.0, std=1.0):
        if hasattr(layer, 'weight'):
            # For uniform distribution
            # init.uniform_(layer.weight, a=lower_bound, b=upper_bound)

            # For normal distribution with larger values
            init.normal_(layer.weight, mean=mean, std=std)

        if hasattr(layer, 'bias') and layer.bias is not None:
            init.constant_(layer.bias, 0.0)


    #Ensure high does not go above maximum bitwidth
    def generate_inputs(self, low=-10, high=10):
        return torch.rand((1, self.in_features)) * (high - low) + low

    def preprocess_tensor(self, tensor, quantizer, parallelism):
        tensor = quantizer(tensor).int()
        logger.info(f"Tensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    def convert_to_integer_list(self,list_val):
        new_list = []
        for val in list_val:
            new_list.append(val.signed_integer)

        return new_list

    async def run_test(self):
        await self.reset()

        logger.info(f"Reset finished")
        self.data_out_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.LLMint_model(inputs)

        # Load the inputs driver
        logger.info(f"Processing inputs")
        inputs = self.preprocess_tensor(
            inputs,
            self.linear_high.x_quantizer,
            int(self.dut.TENSOR_SIZE_DIM),
        )

        self.data_in_driver.load_driver(inputs)

        # Load the weights driver
        logger.info(f"Processing weights")
        logger.info(f"High weights: {self.linear_high.weight}")
        logger.info(f"Low weights: {self.linear_low.weight}")

        weights = self.preprocess_tensor(
            self.linear_high.weight,
            self.linear_high.w_quantizer,
            int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM)
        )

        q_weights = self.preprocess_tensor(
            self.linear_low.weight,
            self.linear_low.w_quantizer,
            int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM)
        )

        self.weight_driver.load_driver(weights)
        self.q_weight_driver.load_driver(q_weights)

        # Load the output monitor
        logger.info(f"Processing outputs: {exp_out}")
        # To do: need to quantize output to a different precision
        outs = self.preprocess_tensor(
            exp_out,
            self.linear_high.x_quantizer,
            int(self.dut.TENSOR_SIZE_DIM)
        )
        self.data_out_monitor.load_monitor(outs)

        await Timer(10000, units="ns")
        logger.info(f"high_precision_masked: {self.convert_to_integer_list(self.dut.high_precision_masked.value)}")
        logger.info(f"low_precision_masked: {self.convert_to_integer_list(self.dut.low_precision_masked.value)}")


        logger.info(f"input_linear_low_precision: {self.convert_to_integer_list(self.dut.input_linear_low_precision.value)}")
        logger.info(f"output_linear_low_precision: {self.convert_to_integer_list(self.dut.output_linear_low_precision.value)}")



        # logger.info(f"input_linear_high_precision: {self.dut.input_linear_high_precision.value}")
        logger.info(f"output_linear_high_precision: {self.convert_to_integer_list(self.dut.output_linear_high_precision.value)}")


        assert self.data_out_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    print('----------------Started test---------------')

    # for i in range(10):
    tb = LinearTB(dut)
    await tb.run_test()


if __name__ == "__main__":


    mase_runner(
        trace=True,
        module_param_list=[
            {
                "ORIGINAL_PRECISION": 16,
                "REDUCED_PRECISION": 8, 
                "TENSOR_SIZE_DIM": global_tensor_size,
                "WEIGHT_DIM_0": global_tensor_size,
                "WEIGHT_DIM_1": global_tensor_size,
                "HIGH_SLOTS": 3,
                "THRESHOLD": 6,
            }],
    )
