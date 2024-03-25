#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t
from functools import partial

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

global_tensor_size = 6


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



        print(type(self.in_features))
        print(type(self.out_features))
        print(type(self.high_slots ))
        print(type(self.threshold ))
        print(type(self.bitwidth ))
        print(type(self.reduced_bitwidth))
        print(type(self.weights_size))

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_driver = StreamDriver(
            dut.clk, dut.data_in, dut.data_in_valid, dut.data_in_ready
        )

        self.weight_driver = StreamDriver(
            dut.clk, dut.weights, dut.weight_valid, dut.weight_ready
        )

        self.quantizer = partial(
            integer_quantizer, width=self.bitwidth, frac_width=0
        )

        self.reduced_quantizer = partial(
            integer_quantizer, width=self.reduced_bitwidth, frac_width=0
        )

        # For latter if we tackle bias
        '''
        if int(dut.HAS_BIAS) == 1:
            self.bias_driver = StreamDriver(
                dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
            )
        '''

        self.data_out_monitor = StreamMonitor(
            dut.clk,
            dut.data_out,
            dut.data_out_valid,
            dut.data_out_ready,
            check=False,
        )

        self.linear_low = LinearInteger(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False,
            config={
                "data_in_width": self.bitwidth,
                "data_in_frac_width": 1,
                "weight_width": self.bitwidth,
                "weight_frac_width": 1,
            },
        )

        self.linear_high = LinearInteger(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False,
            config={
                "data_in_width": self.reduced_bitwidth,
                "data_in_frac_width": 1,
                "weight_width": self.reduced_bitwidth,
                "weight_frac_width": 1,
            },
        )

        # Not sure about this line
        self.linear_low.weight = self.reduced_quantizer(self.linear_high.weight)


    def gather(self, x_low, x_high):
        return x_low + x_high
    
    def scatter(self, x):
        high_mat = []
        low_mat = []
        count_high = 0

        for k in reversed(x.tolist()):
            if abs(k) > self.threshold and count_high < self.high_slots:
                high_mat.append(k)
                low_mat.append(0)
                count_high += 1
            else:
                high_mat.append(0)
                low_mat.append(k)

        low_mat.reverse()
        high_mat.reverse()

        return torch.tensor(low_mat), torch.tensor(high_mat)
    
    def LLMint_model(self, inputs):
        x_low_i, x_high_i = self.scatter(inputs)

        x_low_o = self.linear_low(x_low_i)
        x_high_o = self.linear_high(x_high_i)

        outputs = self.gather(x_low_o, x_high_o)

        return outputs

    def generate_inputs(self):
        return torch.randn((1, self.in_features))

    def preprocess_tensor(self, tensor, quantizer, parallelism):
        tensor = quantizer(tensor).int()
        logger.info(f"Tensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

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
            self.quantizer,
            int(self.dut.TENSOR_SIZE_DIM),
        )
        self.data_in_driver.load_driver(inputs)

        # Load the weights driver
        logger.info(f"Processing weights")
        weights = self.preprocess_tensor(
            self.weights,
            self.quantizer,
            int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM),
        )
        reduced_weights = self.preprocess_tensor(
            self.q_weights,
            self.reduced_quantizer,
            int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM),
        )
        # Combine the weights. weights and reduced_weights are lists of tensors. We need to combine them into 
        # a single list of augmented tensors
        combined_weights = [weights[i] + reduced_weights[i] for i in range(len(weights))]
        self.weight_driver.load_driver(combined_weights)

        # Load the output monitor
        logger.info(f"Processing outputs: {exp_out}")
        # To do: need to quantize output to a different precision
        outs = self.preprocess_tensor(
            exp_out,
            self.quantizer,
            int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0),
        )
        self.data_out_monitor.load_monitor(outs)

        await Timer(1000, units="us")
        assert self.data_out_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
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
            }
        ],
    )