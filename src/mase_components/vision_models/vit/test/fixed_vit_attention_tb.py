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
    ViTAttentionInteger,
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
        self.query_weight_driver = StreamDriver(
            dut.clk, dut.query_weight, dut.query_weight_valid, dut.query_weight_ready
        )
        self.key_weight_driver = StreamDriver(
            dut.clk, dut.key_weight, dut.key_weight_valid, dut.key_weight_ready
        )
        self.value_weight_driver = StreamDriver(
            dut.clk, dut.value_weight, dut.value_weight_valid, dut.value_weight_ready
        )
        self.proj_weight_driver = StreamDriver(
            dut.clk, dut.proj_weight, dut.proj_weight_valid, dut.proj_weight_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.query_bias_driver = StreamDriver(
                dut.clk, dut.query_bias, dut.query_bias_valid, dut.query_bias_ready
            )
            self.key_bias_driver = StreamDriver(
                dut.clk, dut.key_bias, dut.key_bias_valid, dut.key_bias_ready
            )
            self.value_bias_driver = StreamDriver(
                dut.clk, dut.value_bias, dut.value_bias_valid, dut.value_bias_ready
            )
            self.proj_bias_driver = StreamDriver(
                dut.clk, dut.proj_bias, dut.proj_bias_valid, dut.proj_bias_ready
            )
            self.query_bias_driver.log.setLevel(logging.DEBUG)
            self.key_bias_driver.log.setLevel(logging.DEBUG)
            self.value_bias_driver.log.setLevel(logging.DEBUG)
            self.proj_bias_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False,
        )

        # Model
        self.q_config = {
            "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            "qkv_weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
            "qkv_weight_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
            "qkv_bias_width": self.get_parameter("BIAS_PRECISION_0"),
            "qkv_bias_frac_width": self.get_parameter("BIAS_PRECISION_1"),
            "qkv_width": self.get_parameter("QKV_PRECISION_0"),
            "qkv_frac_width": self.get_parameter("QKV_PRECISION_1"),
            "qkmm_out_width": self.get_parameter("QKMM_OUT_PRECISION_0"),
            "qkmm_out_frac_width": self.get_parameter("QKMM_OUT_PRECISION_1"),
            "softmax_exp_width": self.get_parameter("SOFTMAX_EXP_PRECISION_0"),
            "softmax_exp_frac_width": self.get_parameter("SOFTMAX_EXP_PRECISION_1"),
            "softmax_out_frac_width": self.get_parameter(
                "SOFTMAX_OUT_DATA_PRECISION_1"
            ),
            "svmm_out_width": self.get_parameter("SVMM_OUT_PRECISION_0"),
            "svmm_out_frac_width": self.get_parameter("SVMM_OUT_PRECISION_1"),
            "proj_weight_width": self.get_parameter("WEIGHT_PROJ_PRECISION_0"),
            "proj_weight_frac_width": self.get_parameter("WEIGHT_PROJ_PRECISION_1"),
            "proj_bias_width": self.get_parameter("BIAS_PROJ_PRECISION_0"),
            "proj_bias_frac_width": self.get_parameter("BIAS_PROJ_PRECISION_1"),
            "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "data_out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
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
        self.query_weight_driver.log.setLevel(logging.INFO)
        self.key_weight_driver.log.setLevel(logging.DEBUG)
        self.value_weight_driver.log.setLevel(logging.DEBUG)
        self.proj_weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, batches=1, us=100):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for _ in range(batches):
            inputs = self.generate_inputs()
            exp_out = self.model(inputs)[0]

            # * Load the inputs driver
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
            # self.log.info(f"Processing inputs: {inputs}")
            self.data_in_0_driver.load_driver(inputs)

            # * Load the qkv weight driver

            for projection in ["query", "key", "value"]:
                layer = getattr(self.model, f"{projection}")
                weights = (
                    layer.weight.transpose(0, 1)
                    if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1
                    else layer.weight
                )
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
                # self.log.info(f"Processing {projection} weights: {weights}")
                getattr(self, f"{projection}_weight_driver").load_driver(weights)

                # * Load the bias driver
                if self.get_parameter("HAS_BIAS") == 1:
                    bias = getattr(self.model, f"{projection}").bias
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
                    # self.log.info(f"Processing {projection} bias: {bias}")
                    getattr(self, f"{projection}_bias_driver").load_driver(bias)

            # * Load the proj weight driver
            if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
                proj_weight = self.model.proj.weight.transpose(0, 1)
            else:
                proj_weight = self.model.proj.weight
            proj_bias = self.model.proj.bias
            # self.log.info(f"Processing projection weights: {proj_weight}")
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
            self.proj_weight_driver.load_driver(proj_weight)

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
                self.proj_bias_driver.load_driver(proj_bias)
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
            count = [0]
            cocotb.scheduler.add(check_signal(count, self.dut, self.log))
            self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = FixedSelfAttentionTB(dut)
    await tb.run_test(batches=1, us=400)


async def check_signal(count, dut, log):
    while True:
        await RisingEdge(dut.clk)
        handshake_signal_check(
            count,
            dut.head_out_valid,
            dut.head_out_ready,
            dut.value,
            log,
        )


def handshake_signal_check(count, valid, ready, signal, log):
    svalue = [i.signed_integer for i in signal.value]
    if valid.value[0] & ready.value[0]:
        count[0]+=1
        log.debug(f"handshake {signal} count= {count}")



default_config = {
    "NUM_HEADS": 3,
    "HAS_BIAS": 1,
    "WEIGHTS_PRE_TRANSPOSED": 1,
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 48,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 32,
    "DATA_IN_0_PARALLELISM_DIM_0": 4,
    "DATA_IN_0_PARALLELISM_DIM_1": 1,
    "WEIGHT_TENSOR_SIZE_DIM_0": 48,
    "WEIGHT_TENSOR_SIZE_DIM_1": 48,
    "WEIGHT_PARALLELISM_DIM_0": 8,
    "WEIGHT_PARALLELISM_DIM_1": 4,
    "WEIGHT_PROJ_TENSOR_SIZE_DIM_0": 48,
    "WEIGHT_PROJ_TENSOR_SIZE_DIM_1": 48,
    "WEIGHT_PROJ_PARALLELISM_DIM_0": 4,
    "WEIGHT_PROJ_PARALLELISM_DIM_1": 8,
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
#     "NUM_HEADS": 4,
#     "ACTIVATION": 1,
#     "HAS_BIAS": 1,
#     "WEIGHTS_PRE_TRANSPOSED": 1,
#     "DATA_IN_0_TENSOR_SIZE_DIM_0": 128,
#     "DATA_IN_0_TENSOR_SIZE_DIM_1": 64,
#     "DATA_IN_0_PARALLELISM_DIM_0": 4,
#     "DATA_IN_0_PARALLELISM_DIM_1": 2,
#     "WEIGHT_TENSOR_SIZE_DIM_0": 128,
#     "WEIGHT_TENSOR_SIZE_DIM_1": 128,
#     "WEIGHT_PARALLELISM_DIM_0": 2,
#     "WEIGHT_PARALLELISM_DIM_1": 4,

#     "WEIGHT_PROJ_TENSOR_SIZE_DIM_0": 128,
#     "WEIGHT_PROJ_TENSOR_SIZE_DIM_1": 128,
#     "WEIGHT_PROJ_PARALLELISM_DIM_0": 4,
#     "WEIGHT_PROJ_PARALLELISM_DIM_1": 2,

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
#     "SVMM_OUT_PRECISION_0": 10,
#     "SVMM_OUT_PRECISION_1": 4,
#     "WEIGHT_PROJ_PRECISION_0": 16,
#     "WEIGHT_PROJ_PRECISION_1": 8,
#     "BIAS_PROJ_PRECISION_0": 16,
#     "BIAS_PROJ_PRECISION_1": 8,
#     "DATA_OUT_0_PRECISION_0": 10,
#     "DATA_OUT_0_PRECISION_1": 4,
# }
MULT_DATA = 1 / math.sqrt(
    default_config["DATA_IN_0_TENSOR_SIZE_DIM_0"] // default_config["NUM_HEADS"]
)


def get_config(kwargs={}):
    config = default_config
    config.update(kwargs)
    return config


torch.manual_seed(1)



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
    mase_runner(
        trace=True,
        module_param_list=[
            get_config(),
            # get_config(),
        ],
        skip_build=False,
    )


torch.manual_seed(0)
if __name__ == "__main__":
    test_fixed_linear_smoke()
