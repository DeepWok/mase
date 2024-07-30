#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge
from pathlib import Path

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import ViTSelfAttentionHeadInteger
from chop.nn.quantizers import integer_quantizer, integer_floor_quantizer

from mase_components.helper import generate_memory

import pytest
import math
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
            check=True,
        )

        # Model
        self.head_size = self.get_parameter("IN_DATA_TENSOR_SIZE_DIM_0")

        
        self.q_config = {
            "query_width":self.get_parameter("IN_DATA_PRECISION_0"),
            "query_frac_width":self.get_parameter("IN_DATA_PRECISION_1"),
            "key_width":self.get_parameter("IN_DATA_PRECISION_0"),
            "key_frac_width":self.get_parameter("IN_DATA_PRECISION_1"),
            "value_width":self.get_parameter("IN_DATA_PRECISION_0"),
            "value_frac_width":self.get_parameter("IN_DATA_PRECISION_1"),
            "qkmm_out_width":self.get_parameter("QKMM_OUT_PRECISION_0"),
            "qkmm_out_frac_width":self.get_parameter("QKMM_OUT_PRECISION_1"),
            "softmax_exp_width":self.get_parameter("SOFTMAX_EXP_PRECISION_0"),
            "softmax_exp_frac_width":self.get_parameter("SOFTMAX_EXP_PRECISION_1"),
            "softmax_out_frac_width":self.get_parameter("SOFTMAX_OUT_DATA_PRECISION_1"),
            "svmm_out_width":self.get_parameter("OUT_DATA_PRECISION_0"),
            "svmm_out_frac_width":self.get_parameter("OUT_DATA_PRECISION_1"),
        }
        self.model = ViTSelfAttentionHeadInteger(
            dim=self.get_parameter("IN_DATA_TENSOR_SIZE_DIM_0"),
            num_heads=1,
            q_config=self.q_config,
            floor=True,
        )
        # assert self.model.mult_data == torch.tensor(MULT_DATA), f"running set mult data {self.model.mult_data} != {MULT_DATA}"
        # Set verbosity of driver and monitor loggers to debug
        self.query_driver.log.setLevel(logging.DEBUG)
        self.key_driver.log.setLevel(logging.DEBUG)
        self.value_driver.log.setLevel(logging.DEBUG)
        self.out_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, seq_len=1):
        return {
            "query_layer": torch.randn((seq_len, self.head_size)),
            "key_layer": torch.randn((seq_len, self.head_size)),
            "value_layer": torch.randn((seq_len, self.head_size)),
        }

    def preprocess_tensor(self, tensor, config, parallelism, floor=True):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)

        # Quantize
        base_quantizer = integer_floor_quantizer if floor else integer_quantizer
        quantizer = partial(base_quantizer, **config)
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
        # breakpoint()
        parallelism = [
            self.get_parameter("IN_DATA_PARALLELISM_DIM_1"),
            self.get_parameter("IN_DATA_PARALLELISM_DIM_0"),
        ]

        # * Load the query driver
        self.log.info(f"Processing query inputs: {inputs['query_layer']}")
        query_inputs = self.preprocess_tensor(
            tensor=inputs["query_layer"],
            config={
                "width":self.q_config["query_width"],
                "frac_width":self.q_config["query_frac_width"]
            },
            parallelism=parallelism,
        )
        self.query_driver.load_driver(query_inputs)

        # * Load the key driver
        self.log.info(f"Processing key inputs: {inputs['key_layer']}")
        key_inputs = self.preprocess_tensor(
            tensor=inputs["key_layer"],
            config={
                "width":self.q_config["key_width"],
                "frac_width":self.q_config["key_frac_width"]
            },
            parallelism=parallelism,
        )
        self.key_driver.load_driver(key_inputs)

        # * Load the value driver
        self.log.info(f"Processing value inputs: {inputs['value_layer']}")
        value_inputs = self.preprocess_tensor(
            tensor=inputs["value_layer"],
            config={
                "width":self.q_config["value_width"],
                "frac_width":self.q_config["value_frac_width"]
            },
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

        cocotb.start_soon(check_signal(self.dut, self.log))
        await Timer(1, units="ms")
        if not self.out_monitor.exp_queue.empty():
            raise RuntimeError(
                "Reached the end of the test, but the output monitor is not empty."
            )


@cocotb.test()
async def cocotb_test(dut):
    tb = FixedSelfAttentionHeadTB(dut)
    await tb.run_test()

default_config = {
        "IN_DATA_TENSOR_SIZE_DIM_0": 128,
        "IN_DATA_TENSOR_SIZE_DIM_1": 64,
        "IN_DATA_PARALLELISM_DIM_0": 4,
        "IN_DATA_PARALLELISM_DIM_1": 2,
        "IN_DATA_PRECISION_0": 8,
        "IN_DATA_PRECISION_1": 4,
        "ACTIVATION": 1,
        "QKMM_OUT_PRECISION_0": 8,
        "QKMM_OUT_PRECISION_1": 4,
        "SOFTMAX_EXP_PRECISION_0": 16,
        "SOFTMAX_EXP_PRECISION_1": 4,

        "SOFTMAX_OUT_DATA_PRECISION_1": 7,

        "OUT_DATA_PRECISION_0": 12,
        "OUT_DATA_PRECISION_1": 4,
    }
# default_config = {
#         "IN_DATA_TENSOR_SIZE_DIM_0": 4,
#         "IN_DATA_TENSOR_SIZE_DIM_1": 2,
#         "IN_DATA_PARALLELISM_DIM_0": 2,
#         "IN_DATA_PARALLELISM_DIM_1": 1,
#         "ACTIVATION": 1,
#         "IN_DATA_PRECISION_0": 8,
#         "IN_DATA_PRECISION_1": 4,
#         "QKMM_OUT_PRECISION_0": 8,
#         "QKMM_OUT_PRECISION_1": 4,
#         "SOFTMAX_EXP_PRECISION_0": 16,
#         "SOFTMAX_EXP_PRECISION_1": 4,

#         "SOFTMAX_OUT_DATA_PRECISION_1": 7,

#         "OUT_DATA_PRECISION_0": 12,
#         "OUT_DATA_PRECISION_1": 4,
#     }
def get_fixed_self_attention_head_config(kwargs={}):
    config = default_config
    config.update(kwargs)
    return config

torch.manual_seed(1)
async def check_signal(dut, log):
    while True:
        await RisingEdge(dut.clk)
        handshake_signal_check(
            dut.attention_scores_valid, 
            dut.attention_scores_ready, 
            dut.attention_scores, log)
        # handshake_signal_check(dut.rolled_k_valid, dut.rolled_k_ready, dut.rolled_k, log)
        # handshake_signal_check(dut.bias_valid,
        #                        dut.bias_ready,
        #                        dut.bias, log)


def handshake_signal_check(valid, ready, signal, log):
    svalue = [i.signed_integer for i in signal.value]
    if valid.value & ready.value:
        log.debug(f"handshake {signal} = {svalue}")


MULT_DATA = 1 / math.sqrt(default_config["IN_DATA_TENSOR_SIZE_DIM_0"])
@pytest.mark.dev
def test_fixed_self_attention_head_smoke():
    """
    Some quick tests to check if the module is working.
    """

    # * Generate exponential LUT for softmax
    generate_memory.generate_sv_lut(
        "exp",
        default_config["QKMM_OUT_PRECISION_0"],
        default_config["QKMM_OUT_PRECISION_1"],
        default_config["SOFTMAX_EXP_PRECISION_0"],
        default_config["SOFTMAX_EXP_PRECISION_1"],
        path=Path(__file__).parents[1] / "rtl",
        constant_mult=MULT_DATA,
        floor=True,
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
