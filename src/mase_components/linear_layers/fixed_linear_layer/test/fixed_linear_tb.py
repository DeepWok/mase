#!/usr/bin/env python3

import os, pytest

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    ErrorThresholdStreamMonitor,
)
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized.modules.linear import LinearInteger
from chop.nn.quantizers import integer_quantizer


class LinearTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

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

        # self.data_out_0_monitor = StreamMonitor(
        #     dut.clk,
        #     dut.data_out_0,
        #     dut.data_out_0_valid,
        #     dut.data_out_0_ready,
        #     check=True,
        # )

        self.data_out_0_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            width=self.get_parameter("DATA_OUT_0_PRECISION_0"),
            signed=True,
            error_bits=1,
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
        self.log.debug(f"Quantized tensor: {q_tensor}")

        # Convert to integer format
        q_tensor = (q_tensor * 2 ** config["frac_width"]).int()
        self.log.debug(f"Tensor in integer format: {q_tensor}")

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

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
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
        if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
            weights = self.model.weight.transpose(0, 1)
        else:
            weights = self.model.weight

        self.log.info(f"Processing weights: {weights}")
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
            self.log.info(f"Processing bias: {bias}")
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
        self.log.info(f"Processing outputs: {exp_out}")
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


def get_fixed_linear_config(kwargs={}):
    config = {
        "HAS_BIAS": 0,
        "WEIGHTS_PRE_TRANSPOSED": 1,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
        "DATA_IN_0_PARALLELISM_DIM_0": 4,
        "DATA_IN_0_PARALLELISM_DIM_1": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": 20,
        "WEIGHT_TENSOR_SIZE_DIM_1": 20,
        "WEIGHT_PARALLELISM_DIM_0": 4,
        "WEIGHT_PARALLELISM_DIM_1": 4,
        "BIAS_TENSOR_SIZE_DIM_0": 20,
        "BIAS_PARALLELISM_DIM_0": 4,
    }
    config.update(kwargs)
    return config


@pytest.mark.dev
def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_linear_config(),
            get_fixed_linear_config({"WEIGHTS_PRE_TRANSPOSED": 0}),
            # TODO: fix these two cases
            # get_fixed_linear_config({"HAS_BIAS": 1}),
            # get_fixed_linear_config({"HAS_BIAS": 1, "WEIGHTS_PRE_TRANSPOSED": 0}),
        ],
    )


@pytest.mark.dev
def test_fixed_linear_regression():
    """
    More extensive tests to check realistic parameter sizes.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_linear_config(
                {
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 768,
                    "DATA_IN_0_PARALLELISM_DIM_0": 32,
                    "WEIGHT_TENSOR_SIZE_DIM_0": 768,
                    "WEIGHT_TENSOR_SIZE_DIM_1": 768,
                    "WEIGHT_PARALLELISM_DIM_0": 32,
                    "WEIGHT_PARALLELISM_DIM_1": 32,
                    "BIAS_TENSOR_SIZE_DIM_0": 768,
                    "BIAS_PARALLELISM_DIM_0": 32,
                }
            ),
            get_fixed_linear_config(
                {
                    "HAS_BIAS": 1,
                    "WEIGHTS_PRE_TRANSPOSED": 0,
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 768,
                    "DATA_IN_0_PARALLELISM_DIM_0": 32,
                    "WEIGHT_TENSOR_SIZE_DIM_0": 768,
                    "WEIGHT_TENSOR_SIZE_DIM_1": 768,
                    "WEIGHT_PARALLELISM_DIM_0": 32,
                    "WEIGHT_PARALLELISM_DIM_1": 32,
                    "BIAS_TENSOR_SIZE_DIM_0": 768,
                    "BIAS_PARALLELISM_DIM_0": 32,
                }
            ),
        ],
    )


if __name__ == "__main__":
    test_fixed_linear_smoke()
    # test_fixed_linear_regression()
