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

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer

from functools import partial

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

# Set logging verbosity to debug
# os.environ["COCOTB_LOG_LEVEL"] = "DEBUG"


class LinearTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_driver = StreamDriver(
                dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
            )
            self.bias_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        # Model
        self.model = LinearInteger(
            in_features=self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            out_features=self.get_parameter("DATA_OUT_0_TENSOR_SIZE_DIM_0"),
            bias=True if self.get_parameter("HAS_BIAS") == 1 else False,
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "weight_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "bias_width": self.get_parameter("BIAS_PRECISION_0"),
                "bias_frac_width": self.get_parameter("BIAS_PRECISION_1"),
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        return torch.randn((1, self.model.in_features))

    def preprocess_tensor(self, tensor, config, parallelism):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)

        # Quantize
        quantizer = partial(integer_quantizer, **config)
        q_tensor = quantizer(tensor)
        logger.debug(f"Quantized tensor: {q_tensor}")

        # Convert to integer format
        q_tensor = (q_tensor * 2 ** config["frac_width"]).int()
        logger.debug(f"Tensor in integer format: {q_tensor}")

        # Split into chunks according to parallelism in each dimension
        # parallelism[0]: along rows, parallelism[1]: along columns
        dim_0_split = q_tensor.split(parallelism[0], dim=0)
        dim_1_split = [x.split(parallelism[1], dim=1) for x in dim_0_split]
        blocks = []
        # Flatten the list of blocks
        for i in range(len(dim_1_split)):
            for j in range(len(dim_1_split[i])):
                blocks.append(dim_1_split[i][j].flatten().tolist())
        return blocks

    def get_parameter(self, parameter_name):
        parameter = getattr(self.dut, parameter_name)
        return int(parameter)

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        # * Load the inputs driver
        logger.info(f"Processing inputs: {inputs}")
        inputs = self.preprocess_tensor(
            tensor=inputs,
            config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_in_0_driver.load_driver(inputs)

        # * Load the weights driver
        weights = self.model.weight.transpose(0, 1)
        logger.info(f"Processing weights: {weights}")
        weights = self.preprocess_tensor(
            tensor=weights,
            config={
                "width": self.get_parameter("WEIGHT_PRECISION_0"),
                "frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("WEIGHT_PARALLELISM_DIM_1"),
                self.get_parameter("WEIGHT_PARALLELISM_DIM_0"),
            ],
        )
        self.weight_driver.load_driver(weights)

        # * Load the bias driver
        if self.get_parameter("HAS_BIAS") == 1:
            bias = self.model.bias
            logger.info(f"Processing bias: {bias}")
            bias = self.preprocess_tensor(
                tensor=bias,
                config={
                    "width": self.get_parameter("BIAS_PRECISION_0"),
                    "frac_width": self.get_parameter("BIAS_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                    self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                ],
            )
            self.bias_driver.load_driver(bias)

        # * Load the output monitor
        logger.info(f"Processing outputs: {exp_out}")
        outs = self.preprocess_tensor(
            tensor=exp_out,
            config={
                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(1, units="ms")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = LinearTB(dut)
    await tb.run_test()


def test_fixed_linear():
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "HAS_BIAS": 1,
                "DATA_IN_0_PRECISION_0": 16,
                "DATA_IN_0_PRECISION_1": 3,
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 4,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "WEIGHT_PRECISION_0": 16,
                "WEIGHT_PRECISION_1": 3,
                "WEIGHT_TENSOR_SIZE_DIM_0": 20,
                "WEIGHT_TENSOR_SIZE_DIM_1": 20,
                "WEIGHT_PARALLELISM_DIM_0": 4,
                "WEIGHT_PARALLELISM_DIM_1": 4,
                "BIAS_PRECISION_0": 16,
                "BIAS_PRECISION_1": 3,
                "BIAS_TENSOR_SIZE_DIM_0": 20,
                "BIAS_PARALLELISM_DIM_0": 4,
                "BIAS_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_PRECISION_0": 35,
                "DATA_OUT_0_PRECISION_1": 6,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 20,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 4,
            }
        ],
    )


if __name__ == "__main__":
    test_fixed_linear()
