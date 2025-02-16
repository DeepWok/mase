#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer
from pathlib import Path

from transformers.models.bert.configuration_bert import BertConfig

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import BertSelfAttentionHeadInteger
from chop.nn.quantizers import integer_quantizer

from mase_components.helper import generate_memory

import pytest


class FixedSelfAttentionHeadTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # * QKV drivers
        self.query_driver = StreamDriver(
            dut.clk, dut.query, dut.query_valid, dut.query_ready
        )
        self.key_driver = StreamDriver(dut.clk, dut.key, dut.key_valid, dut.key_ready)
        self.value_driver = StreamDriver(
            dut.clk, dut.value, dut.value_valid, dut.value_ready
        )

        self.out_monitor = StreamMonitor(
            dut.clk,
            dut.out,
            dut.out_valid,
            dut.out_ready,
            check=False,
        )

        # Model
        self.config = BertConfig()
        self.head_size = self.config.hidden_size // self.config.num_attention_heads

        self.q_config = {
            "width": self.get_parameter("IN_DATA_PRECISION_0"),
            "frac_width": self.get_parameter("IN_DATA_PRECISION_1"),
        }
        self.model = BertSelfAttentionHeadInteger(
            config=self.config,
            q_config=self.q_config,
        )

        # Set verbosity of driver and monitor loggers to debug
        # self.query_driver.log.setLevel(logging.DEBUG)
        # self.key_driver.log.setLevel(logging.DEBUG)
        # self.value_driver.log.setLevel(logging.DEBUG)
        # self.out_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, seq_len=20):
        return {
            "query_layer": torch.randn((seq_len, self.head_size)),
            "key_layer": torch.randn((seq_len, self.head_size)),
            "value_layer": torch.randn((seq_len, self.head_size)),
        }

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
        self.out_monitor.ready.value = 1

        inputs = self.generate_inputs(
            seq_len=self.get_parameter("IN_DATA_TENSOR_SIZE_DIM_1")
        )
        exp_out = self.model(**inputs)

        parallelism = [
            self.get_parameter("IN_DATA_PARALLELISM_DIM_1"),
            self.get_parameter("IN_DATA_PARALLELISM_DIM_0"),
        ]

        # * Load the query driver
        self.log.info(f"Processing query inputs: {inputs['query_layer']}")
        query_inputs = self.preprocess_tensor(
            tensor=inputs["query_layer"],
            config=self.q_config,
            parallelism=parallelism,
        )
        self.query_driver.load_driver(query_inputs)

        # * Load the key driver
        self.log.info(f"Processing key inputs: {inputs['key_layer']}")
        key_inputs = self.preprocess_tensor(
            tensor=inputs["key_layer"],
            config=self.q_config,
            parallelism=parallelism,
        )
        self.key_driver.load_driver(key_inputs)

        # * Load the value driver
        self.log.info(f"Processing value inputs: {inputs['value_layer']}")
        value_inputs = self.preprocess_tensor(
            tensor=inputs["value_layer"],
            config=self.q_config,
            parallelism=parallelism,
        )
        self.value_driver.load_driver(value_inputs)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        outs = self.preprocess_tensor(
            tensor=exp_out,
            config={
                "width": self.get_parameter("OUT_DATA_PRECISION_0"),
                "frac_width": self.get_parameter("OUT_DATA_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("OUT_DATA_PARALLELISM_DIM_1"),
                self.get_parameter("OUT_DATA_PARALLELISM_DIM_0"),
            ],
        )
        self.out_monitor.load_monitor(outs)

        await Timer(1, units="ms")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError(
                "Reached the end of the test, but the output monitor is not empty."
            )


@cocotb.test()
async def cocotb_test(dut):
    tb = FixedSelfAttentionHeadTB(dut)
    await tb.run_test()


def get_fixed_self_attention_head_config(kwargs={}):
    config = {
        "IN_DATA_TENSOR_SIZE_DIM_0": 64,
        "IN_DATA_TENSOR_SIZE_DIM_1": 32,
        "IN_DATA_PARALLELISM_DIM_0": 2,
        "IN_DATA_PARALLELISM_DIM_1": 2,
        "IN_DATA_PRECISION_0": 16,
        "IN_DATA_PRECISION_1": 3,
        "OUT_DATA_TENSOR_SIZE_DIM_0": 64,
        "OUT_DATA_TENSOR_SIZE_DIM_1": 32,
        "OUT_DATA_PARALLELISM_DIM_0": 2,
        "OUT_DATA_PARALLELISM_DIM_1": 2,
        "OUT_DATA_PRECISION_0": 16,
        "OUT_DATA_PRECISION_1": 3,
    }
    config.update(kwargs)
    return config


@pytest.mark.dev
def test_fixed_self_attention_head_smoke():
    """
    Some quick tests to check if the module is working.
    """

    # * Generate exponential LUT for softmax
    generate_memory.generate_sv_lut(
        "exp",
        16,
        3,
        16,
        3,
        path=Path(__file__).parents[1] / "rtl",
    )
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_self_attention_head_config(),
        ],
        skip_build=False,
    )


if __name__ == "__main__":
    test_fixed_self_attention_head_smoke()
