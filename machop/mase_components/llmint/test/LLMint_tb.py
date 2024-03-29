#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
import numpy as np
from functools import partial

import cocotb
from cocotb.binary import BinaryValue
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
logging.getLogger().setLevel(logging.WARNING)


class LLMStreamMonitor(StreamMonitor):
    def _recv(self):
        if type(self.data.value) == list:
            return [x for x in self.data.value]
        elif type(self.data.value) == BinaryValue:
            return int(self.data.value)

    def _check(self, got, exp):
        if self.check:
            # print("\nGot \n%s, \nExpected \n%s" % (self.convert_to_integer_list(got), exp))
            assert np.equal(self.convert_to_integer_list(got), exp).all()

    def convert_to_integer_list(self, list_val):
        new_list = []
        for val in list_val:
            new_list.append(val.signed_integer)

        return new_list


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

        # ----------------Drivers---------------
        # print(self.in_features)
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
        """
        if int(dut.HAS_BIAS) == 1:
            self.bias_driver = StreamDriver(
                dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
            )
        """
        # ----------------Monitor B---------------

        self.data_out_monitor = LLMStreamMonitor(
            dut.clk,
            dut.data_out,
            dut.data_out_valid,
            dut.data_out_ready,
            check=True,
        )

        # ----------------Linear low---------------

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

        # ----------------Linear high--------------

        self.linear_high = LinearInteger(
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

        self.initialize_weights(
            self.linear_high, mean=0.0, std=3.0
        )  # Adjust mean and std as needed for larger values
        self.linear_low.weight = (
            self.linear_high.weight
        )  # Both linear layers should have the same weights

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

        # print('low_mat',low_mat)
        # print('high_mat',high_mat)

        return torch.tensor(low_mat), torch.tensor(high_mat)

    def LLMint_model(self, inputs):
        x_low_i, x_high_i = self.scatter(inputs)

        x_low_i = self.linear_low.x_quantizer(x_low_i).int()

        x_low_o = self.linear_low(x_low_i)
        x_high_o = self.linear_high(x_high_i)

        outputs = self.gather(x_low_o, x_high_o)

        return outputs

    def initialize_weights(self, layer, mean=0.0, std=1.0):
        if hasattr(layer, "weight"):
            # For uniform distribution
            # init.uniform_(layer.weight, a=lower_bound, b=upper_bound)

            # For normal distribution with larger values
            init.normal_(layer.weight, mean=mean, std=std)

        if hasattr(layer, "bias") and layer.bias is not None:
            init.constant_(layer.bias, 0.0)

    # Ensure high does not go above maximum bitwidth
    def generate_inputs(self, low=-10, high=10):
        return torch.rand((1, self.in_features)) * (high - low) + low

    def preprocess_tensor(self, tensor, quantizer, parallelism):
        tensor = quantizer(tensor).int()
        # logger.info(f"Tensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    def convert_to_integer_list(self, list_val):
        new_list = []
        for val in list_val:
            new_list.append(val.signed_integer)

        return new_list

    async def run_test(self, runs):
        await self.reset()
        for i in range(runs):

            logger.info(f"Reset finished")
            self.data_out_monitor.ready.value = 1

            inputs = self.generate_inputs()
            exp_out = self.LLMint_model(inputs)

            # Load the inputs driver
            # logger.info(f"Processing inputs")
            inputs = self.preprocess_tensor(
                inputs,
                self.linear_high.x_quantizer,
                int(self.dut.TENSOR_SIZE_DIM),
            )

            self.data_in_driver.load_driver(inputs)

            # Load the weights driver
            # logger.info(f"Processing weights")
            # logger.info(f"High weights: {self.linear_high.weight}")
            # logger.info(f"Low weights: {self.linear_low.weight}")

            weights = self.preprocess_tensor(
                self.linear_high.weight,
                self.linear_high.w_quantizer,
                int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM),
            )

            q_weights = self.preprocess_tensor(
                self.linear_low.weight,
                self.linear_low.w_quantizer,
                int(self.dut.TENSOR_SIZE_DIM) * int(self.dut.TENSOR_SIZE_DIM),
            )

            self.weight_driver.load_driver(weights)
            self.q_weight_driver.load_driver(q_weights)

            # Load the output monitor
            # logger.info(f"Processing outputs: {exp_out}")
            # To do: need to quantize output to a different precision
            outs = self.preprocess_tensor(
                exp_out, self.linear_high.x_quantizer, int(self.dut.TENSOR_SIZE_DIM)
            )
            self.data_out_monitor.load_monitor(outs)

            await Timer(1000, units="ns")
        assert self.data_out_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    print("----------------Started test---------------")

    # for i in range(10):
    tb = LinearTB(dut)
    await tb.run_test(200)


def create_params(
    original_precision_values=[16],
    reduced_precision_values=[8, 16],
    tensor_size_dim_values=[8, 16, 32, 64],
    high_slots_values=[2, 4, 8, 16, 32],
    threshold_values=[6, 12, 32],
):  # [4, 8, 16, 32]):
    # Generate all possible combinations
    all_combinations = [
        (op, rp, tsd, hs, th)
        for op in original_precision_values
        for rp in reduced_precision_values
        for tsd in tensor_size_dim_values
        for hs in high_slots_values
        for th in threshold_values
    ]

    # Filter combinations based on ORIGINAL_PRECISION should be greater than REDUCED_PRECISION, and HIGH_SLOTS should be less than half of TENSOR_SIZE_DIM
    filtered_combinations = filter(
        lambda x: x[0] > x[1] and x[2] // 2 > x[3], all_combinations
    )

    # Construct the parameter dictionaries
    module_param_list = [
        {
            "ORIGINAL_PRECISION": op,
            "REDUCED_PRECISION": rp,
            "TENSOR_SIZE_DIM": tsd,
            "WEIGHT_DIM_0": tsd,
            "WEIGHT_DIM_1": tsd,
            "HIGH_SLOTS": hs,
            "THRESHOLD": th,
        }
        for op, rp, tsd, hs, th in filtered_combinations
    ]
    for i in range(len(module_param_list)):
        print(module_param_list[i])

    return module_param_list


if __name__ == "__main__":

    # mase_runner(
    #     trace=True,
    #     module_param_list=create_params()
    # )

    mase_runner(
        trace=True,
        module_param_list=create_params(
            original_precision_values=[16],
            reduced_precision_values=[8],
            tensor_size_dim_values=[16],
            high_slots_values=[4],
        ),
    )
