#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from transformers.models.bert.configuration_bert import BertConfig

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import BertSelfAttentionInteger, fixed_softermax

from mase_cocotb.utils import fixed_preprocess_tensor


class FixedSelfAttentionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        # * Weight drivers
        self.weight_query_driver = StreamDriver(
            dut.clk, dut.weight_query, dut.weight_query_valid, dut.weight_query_ready
        )
        self.weight_key_driver = StreamDriver(
            dut.clk, dut.weight_key, dut.weight_key_valid, dut.weight_key_ready
        )
        self.weight_value_driver = StreamDriver(
            dut.clk, dut.weight_value, dut.weight_value_valid, dut.weight_value_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_query_driver = StreamDriver(
                dut.clk, dut.biasquery_, dut.bias_query_valid, dut.bias_query_ready
            )
            self.bias_key_driver = StreamDriver(
                dut.clk, dut.bias_key, dut.bias_key_valid, dut.bias_key_ready
            )
            self.bias_value_driver = StreamDriver(
                dut.clk, dut.bias_value, dut.bias_value_valid, dut.bias_value_ready
            )
            self.bias_query_driver.log.setLevel(logging.DEBUG)
            self.bias_key_driver.log.setLevel(logging.DEBUG)
            self.bias_value_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False,
        )

        # Model
        self.config = BertConfig()
        self.config.hidden_size = self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0")
        self.config.num_attention_heads = self.get_parameter("NUM_HEADS")
        self.q_config = {
            "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
            "weight_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
            "bias_width": self.get_parameter("BIAS_PRECISION_0"),
            "bias_frac_width": self.get_parameter("BIAS_PRECISION_1"),
        }
        self.out_q_config = {
            "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "data_out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
        }
        self.model = BertSelfAttentionInteger(
            config=self.config,
            q_config=self.q_config,
            out_q_config=self.out_q_config,
            bias=self.get_parameter("HAS_BIAS"),
            floor=True,
        )
        # * Replace softmax with fixed softermax
        if self.get_parameter("ACTIVATION") == 0:
            self.model.softmax = partial(
                fixed_softermax,
                q_config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
                out_q_config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
            )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_query_driver.log.setLevel(logging.DEBUG)
        self.weight_key_driver.log.setLevel(logging.DEBUG)
        self.weight_value_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)[0]

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
        inputs = fixed_preprocess_tensor(
            tensor=inputs,
            q_config={
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
        for projection in ["query", "key", "value"]:

            if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
                weights = getattr(self.model, projection).weight.transpose(0, 1)
            else:
                weights = getattr(self.model, projection).weight

            self.log.info(f"Processing {projection} weights: {weights}")
            weights = fixed_preprocess_tensor(
                tensor=weights,
                q_config={
                    "width": self.get_parameter("WEIGHT_PRECISION_0"),
                    "frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("WEIGHT_PARALLELISM_DIM_1"),
                    self.get_parameter("WEIGHT_PARALLELISM_DIM_0"),
                ],
            )
            getattr(self, f"weight_{projection}_driver").load_driver(weights)

            # * Load the bias driver
            if self.get_parameter("HAS_BIAS") == 1:
                bias = getattr(self.model, projection).bias
                self.log.info(f"Processing {projection} bias: {bias}")
                bias = fixed_preprocess_tensor(
                    tensor=bias,
                    q_config={
                        "width": self.get_parameter("BIAS_PRECISION_0"),
                        "frac_width": self.get_parameter("BIAS_PRECISION_1"),
                    },
                    parallelism=[
                        self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                        self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                    ],
                )
                getattr(self, f"bias_{projection}_driver").load_driver(bias)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        outs = fixed_preprocess_tensor(
            tensor=exp_out,
            q_config={
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
    tb = FixedSelfAttentionTB(dut)
    await tb.run_test()


def get_config(kwargs={}):
    config = {
        "NUM_HEADS": 1,
        "ACTIVATION": 0,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_IN_0_PRECISION_0": 16,
        "DATA_IN_0_PRECISION_1": 8,
        "WEIGHTS_PRE_TRANSPOSED": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": 4,
        "WEIGHT_TENSOR_SIZE_DIM_1": 4,
        "WEIGHT_PARALLELISM_DIM_0": 2,
        "WEIGHT_PARALLELISM_DIM_1": 2,
        "WEIGHT_PRECISION_0": 16,
        "WEIGHT_PRECISION_1": 8,
        "HAS_BIAS": 0,
        "BIAS_TENSOR_SIZE_DIM_0": 4,
        "BIAS_TENSOR_SIZE_DIM_1": 4,
        "BIAS_PARALLELISM_DIM_0": 2,
        "BIAS_PARALLELISM_DIM_1": 2,
        "BIAS_PRECISION_0": 16,
        "BIAS_PRECISION_1": 8,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_OUT_0_PARALLELISM_DIM_0": 2,
        "DATA_OUT_0_PARALLELISM_DIM_1": 2,
        "DATA_OUT_0_PRECISION_0": 16,
        "DATA_OUT_0_PRECISION_1": 8,
    }
    config.update(kwargs)
    return config


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(trace=True, module_param_list=[get_config()], skip_build=False)


if __name__ == "__main__":
    test_fixed_linear_smoke()
