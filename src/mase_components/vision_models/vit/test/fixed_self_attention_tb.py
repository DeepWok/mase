#!/usr/bin/env python3

import os

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge

import math

from mase_components.helper import generate_memory
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import (
    ViTAttentionInteger
)
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
        self.weight_proj_driver = StreamDriver(
            dut.clk, dut.weight_proj, dut.weight_proj_valid, dut.weight_proj_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_query_driver = StreamDriver(
                dut.clk, dut.bias_query, dut.bias_query_valid, dut.bias_query_ready
            )
            self.bias_key_driver = StreamDriver(
                dut.clk, dut.bias_key, dut.bias_key_valid, dut.bias_key_ready
            )
            self.bias_value_driver = StreamDriver(
                dut.clk, dut.bias_value, dut.bias_value_valid, dut.bias_value_ready
            )
            self.bias_proj_driver = StreamDriver(
                dut.clk, dut.bias_proj, dut.bias_proj_valid, dut.bias_proj_ready
            )
            self.bias_query_driver.log.setLevel(logging.DEBUG)
            self.bias_key_driver.log.setLevel(logging.DEBUG)
            self.bias_value_driver.log.setLevel(logging.DEBUG)
            self.bias_proj_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

        # Model
        self.q_config = {
            "data_in_width":self.get_parameter("DATA_IN_0_PRECISION_0"),
            "data_in_frac_width":self.get_parameter("DATA_IN_0_PRECISION_1"),
            "qkv_weight_width":self.get_parameter("WEIGHT_PRECISION_0"),
            "qkv_weight_frac_width":self.get_parameter("WEIGHT_PRECISION_1"),
            "qkv_bias_width":self.get_parameter("BIAS_PRECISION_0"),
            "qkv_bias_frac_width":self.get_parameter("BIAS_PRECISION_1"),
            "qkv_width":self.get_parameter("QKV_PRECISION_0"),
            "qkv_frac_width":self.get_parameter("QKV_PRECISION_1"),
            "qkmm_out_width":self.get_parameter("QKMM_OUT_PRECISION_0"),
            "qkmm_out_frac_width":self.get_parameter("QKMM_OUT_PRECISION_1"),
            "softmax_exp_width":self.get_parameter("SOFTMAX_EXP_PRECISION_0"),
            "softmax_exp_frac_width":self.get_parameter("SOFTMAX_EXP_PRECISION_1"),
            "softmax_out_frac_width":self.get_parameter("SOFTMAX_OUT_DATA_PRECISION_1"),
            "svmm_out_width":self.get_parameter("SVMM_OUT_PRECISION_0"),
            "svmm_out_frac_width":self.get_parameter("SVMM_OUT_PRECISION_1"),

            "proj_weight_width":self.get_parameter("WEIGHT_PROJ_PRECISION_0"),
            "proj_weight_frac_width":self.get_parameter("WEIGHT_PROJ_PRECISION_1"),
            "proj_bias_width":self.get_parameter("BIAS_PROJ_PRECISION_0"),
            "proj_bias_frac_width":self.get_parameter("BIAS_PROJ_PRECISION_1"),
            "data_out_width":self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "data_out_frac_width":self.get_parameter("DATA_OUT_0_PRECISION_1"),
        }
        self.model = ViTAttentionInteger(
                dim=self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
                num_heads=self.get_parameter("NUM_HEADS"),
                qkv_bias=True if self.get_parameter("HAS_BIAS") else False,
                q_config=self.q_config,
                floor=True,
            )


        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_query_driver.log.setLevel(logging.DEBUG)
        self.weight_key_driver.log.setLevel(logging.DEBUG)
        self.weight_value_driver.log.setLevel(logging.DEBUG)
        self.weight_proj_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, us=100):
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
            floor=True,
        )
        self.data_in_0_driver.load_driver(inputs)

        # * Load the qkv weight driver
        qkv_weight = self.model.qkv.weight.reshape(
            3, self.get_parameter("WEIGHT_TENSOR_SIZE_DIM_1"), self.get_parameter("WEIGHT_TENSOR_SIZE_DIM_0"))
        qkv_bias = self.model.qkv.bias.reshape(
            3, self.get_parameter("BIAS_TENSOR_SIZE_DIM_1"), self.get_parameter("BIAS_TENSOR_SIZE_DIM_0"))
        i = 0
        for projection in ["query", "key", "value"]:
            
            if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
                weights = qkv_weight[i].transpose(0, 1)
            else:
                weights = qkv_weight[i]

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
                floor=True,
            )
            getattr(self, f"weight_{projection}_driver").load_driver(weights)

            # * Load the bias driver
            if self.get_parameter("HAS_BIAS") == 1:
                bias = qkv_bias[i]
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
                    floor=True,
                )
                getattr(self, f"bias_{projection}_driver").load_driver(bias)
            i = i+1
        
        # * Load the proj weight driver
        if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
            proj_weight = self.model.proj.weight.transpose(0, 1)
        else:
            proj_weight = self.model.proj.weight
        proj_bias = self.model.proj.bias
        self.log.info(f"Processing projection weights: {proj_weight}")
        proj_weight = fixed_preprocess_tensor(
            tensor=proj_weight,
            q_config={
                "width": self.get_parameter("WEIGHT_PRECISION_0"),
                "frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("WEIGHT_PROJ_PARALLELISM_DIM_1"),
                self.get_parameter("WEIGHT_PROJ_PARALLELISM_DIM_0"),
            ],
            floor=True,
        )
        self.weight_proj_driver.load_driver(proj_weight)

        # * Load the bias driver
        if self.get_parameter("HAS_BIAS") == 1:
            self.log.info(f"Processing projection bias: {proj_bias}")
            proj_bias = fixed_preprocess_tensor(
                tensor=proj_bias,
                q_config={
                    "width": self.get_parameter("BIAS_PROJ_PRECISION_0"),
                    "frac_width": self.get_parameter("BIAS_PROJ_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("BIAS_PROJ_PARALLELISM_DIM_1"),
                    self.get_parameter("BIAS_PROJ_PARALLELISM_DIM_0"),
                ],
                floor=True,
            )
            self.bias_proj_driver.load_driver(proj_bias)
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
            floor=True,
        )
        self.data_out_0_monitor.load_monitor(outs)
        cocotb.start_soon(check_signal(self.dut, self.log))
        
        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = FixedSelfAttentionTB(dut)
    await tb.run_test(us=100)

default_config = {
    "NUM_HEADS": 4,
    "ACTIVATION": 1,
    "HAS_BIAS": 1,
    "WEIGHTS_PRE_TRANSPOSED": 1,
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 16,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
    "DATA_IN_0_PARALLELISM_DIM_0": 4,
    "DATA_IN_0_PARALLELISM_DIM_1": 2,
    "WEIGHT_TENSOR_SIZE_DIM_0": 16,
    "WEIGHT_TENSOR_SIZE_DIM_1": 16,
    "WEIGHT_PARALLELISM_DIM_0": 2,
    "WEIGHT_PARALLELISM_DIM_1": 4,

    "WEIGHT_PROJ_TENSOR_SIZE_DIM_0": 16,
    "WEIGHT_PROJ_TENSOR_SIZE_DIM_1": 16,
    "WEIGHT_PROJ_PARALLELISM_DIM_0": 4,
    "WEIGHT_PROJ_PARALLELISM_DIM_1": 2,
    
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 3,
    "WEIGHT_PRECISION_0": 16,
    "WEIGHT_PRECISION_1": 8,
    "BIAS_PRECISION_0": 16,
    "BIAS_PRECISION_1": 8,
    "QKV_PRECISION_0": 8,
    "QKV_PRECISION_1": 3,
    "QKMM_OUT_PRECISION_0": 8,
    "QKMM_OUT_PRECISION_1": 3,
    "SOFTMAX_EXP_PRECISION_0": 12,
    "SOFTMAX_EXP_PRECISION_1": 4,
    "SOFTMAX_OUT_DATA_PRECISION_1": 6,

    "SVMM_OUT_PRECISION_0": 10,
    "SVMM_OUT_PRECISION_1": 4,
    "WEIGHT_PROJ_PRECISION_0": 16,
    "WEIGHT_PROJ_PRECISION_1": 8,
    "BIAS_PROJ_PRECISION_0": 16,
    "BIAS_PROJ_PRECISION_1": 8,
    "DATA_OUT_0_PRECISION_0": 10,
    "DATA_OUT_0_PRECISION_1": 4,
}
# default_config = {
#     "NUM_HEADS": 2,
#     "ACTIVATION": 1,
#     "HAS_BIAS": 1,
#     "WEIGHTS_PRE_TRANSPOSED": 1,
#     "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
#     "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
#     "DATA_IN_0_PARALLELISM_DIM_0": 2,
#     "DATA_IN_0_PARALLELISM_DIM_1": 1,
#     "WEIGHT_TENSOR_SIZE_DIM_0": 4,
#     "WEIGHT_TENSOR_SIZE_DIM_1": 4,
#     "WEIGHT_PARALLELISM_DIM_0": 1,
#     "WEIGHT_PARALLELISM_DIM_1": 2,
    
#     "DATA_IN_0_PRECISION_0": 8,
#     "DATA_IN_0_PRECISION_1": 3,
#     "WEIGHT_PRECISION_0": 16,
#     "WEIGHT_PRECISION_1": 8,
#     "BIAS_PRECISION_0": 16,
#     "BIAS_PRECISION_1": 8,
#     "QKV_PRECISION_0": 8,
#     "QKV_PRECISION_1": 3,
#     "QKMM_OUT_PRECISION_0": 8,
#     "QKMM_OUT_PRECISION_1": 3,
#     "SOFTMAX_EXP_PRECISION_0": 12,
#     "SOFTMAX_EXP_PRECISION_1": 4,
#     "SOFTMAX_OUT_DATA_PRECISION_1": 6,
#     "DATA_OUT_0_PRECISION_0": 10,
#     "DATA_OUT_0_PRECISION_1": 4,
# }
MULT_DATA = 1 / math.sqrt(default_config["DATA_IN_0_TENSOR_SIZE_DIM_0"] // default_config["NUM_HEADS"])
def get_config(kwargs={}):
    config = default_config
    config.update(kwargs)
    return config

torch.manual_seed(1)
async def check_signal(dut, log):
    while True:
        await RisingEdge(dut.clk)
        # handshake_signal_check(
        #     dut.g_attention_head[0].head_i.query_key_transpose_valid, 
        #     dut.g_attention_head[0].head_i.query_key_transpose_ready, 
        #     dut.g_attention_head[0].head_i.query_key_transpose, log)
        # handshake_signal_check(dut.rolled_k_valid, dut.rolled_k_ready, dut.rolled_k, log)
        # handshake_signal_check(dut.bias_valid,
        #                        dut.bias_ready,
        #                        dut.bias, log)


def handshake_signal_check(valid, ready, signal, log):
    svalue = [i.signed_integer for i in signal.value]
    if valid.value & ready.value:
        log.debug(f"handshake {signal} = {svalue}")



def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    generate_memory.generate_sv_lut(
        "exp",
        default_config["QKMM_OUT_PRECISION_0"],
        default_config["QKMM_OUT_PRECISION_1"],
        default_config["SOFTMAX_EXP_PRECISION_0"],
        default_config["SOFTMAX_EXP_PRECISION_1"],
        constant_mult=MULT_DATA,
        floor=True,
    )
    mase_runner(trace=True, module_param_list=[get_config()], skip_build=False)

torch.manual_seed(0)
if __name__ == "__main__":
    test_fixed_linear_smoke()
