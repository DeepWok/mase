#!/usr/bin/env python3

import os

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


class ConvArithTB(Testbench):
    def __init__(self, dut, number_of_examples=1) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.number_of_examples = number_of_examples

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )

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
        # self.data_out_0_monitor = StreamMonitor(
        #     dut.clk,
        #     dut.data_out_0,
        #     dut.data_out_0_valid,
        #     dut.data_out_0_ready,
        #     width=self.get_parameter("DATA_OUT_0_PRECISION_0"),
        #     signed=True,
        #     error_bits=1,
        #     check=True,
        # )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_data(self, config):
        torch.manual_seed(10)
        roll_in_num = self.get_parameter("ROLL_IN_NUM")
        roll_out_num = self.get_parameter("ROLL_OUT_NUM")
        icd = self.get_parameter("IN_CHANNELS_DEPTH")
        ocp = self.get_parameter("OUT_CHANNELS_PARALLELISM")
        weight_repeats = self.get_parameter("WEIGHT_REPEATS")
        ocd = self.get_parameter("OUT_CHANNELS_DEPTH")
        has_bias = self.get_parameter("HAS_BIAS")
        oc = ocd * ocp
        x = torch.randn((self.number_of_examples, weight_repeats, icd * roll_in_num))
        weight = torch.randn(
            (self.number_of_examples, 1, oc, icd * roll_in_num)
        ).repeat(1, weight_repeats, 1, 1)
        bias = torch.randn(self.number_of_examples, 1, oc).repeat(1, weight_repeats, 1)
        qx = integer_quantizer(x, config["x_width"], config["x_frac_width"])
        qw = integer_quantizer(weight, config["w_width"], config["w_frac_width"])
        qb = integer_quantizer(bias, config["b_width"], config["b_frac_width"])
        out = torch.sum(qx.unsqueeze(2) * qw, dim=-1)

        out = out + qb if has_bias else out

        qx_tensor = (qx * 2 ** config["x_frac_width"]).int()
        qw_tensor = (qw * 2 ** config["w_frac_width"]).int()
        qb_tensor = (qb * 2 ** config["b_frac_width"]).int()

        self.log.info(f"Processing inputs: {qx_tensor}")
        self.log.info(f"Processing weights: {qw_tensor}")
        self.log.info(f"Processing bias: {qb_tensor}")
        block_qx = qx_tensor.reshape(-1, roll_out_num).tolist()
        block_qw = (
            qw_tensor.reshape(
                self.number_of_examples * ocd * weight_repeats, ocp, -1, roll_out_num
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, ocp * roll_out_num)
            .tolist()
        )
        block_qb = qb_tensor.reshape(-1, ocp).tolist()

        # qout = integer_floor_quantizer(out, config["out_width"], config["out_frac_width"])
        qout_tensor = (out * 2 ** config["out_frac_width"]).int()
        self.log.info(f"Processing outs: {qout_tensor}")
        block_outs = qout_tensor.reshape(-1, ocp).tolist()

        return block_qx, block_qw, block_qb, block_outs

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

        x, w, b, o = self.generate_data(
            {
                "x_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "x_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "w_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "w_frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "b_width": self.get_parameter("BIAS_PRECISION_0"),
                "b_frac_width": self.get_parameter("BIAS_PRECISION_1"),
                "out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            }
        )
        # * Load the inputs driver
        self.log.info(f"Processing inputs: {x}")
        self.data_in_0_driver.load_driver(x)

        # * Load the weights driver
        self.log.info(f"Processing weights: {w}")
        self.weight_driver.load_driver(w)

        # * Load the bias driver
        self.log.info(f"Processing bias: {b}")
        self.bias_driver.load_driver(b)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {o}")
        self.data_out_0_monitor.load_monitor(o)

        await Timer(10, units="ms")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = ConvArithTB(dut, 2)
    await tb.run_test()


def get_fixed_linear_config(kwargs={}):
    config = {
        "HAS_BIAS": 1,
        "ROLL_IN_NUM": 1,
        "ROLL_OUT_NUM": 1,
        "IN_CHANNELS_DEPTH": 1,
        "OUT_CHANNELS_PARALLELISM": 2,
        "WEIGHT_REPEATS": 2,
        "OUT_CHANNELS_DEPTH": 1,
    }
    config.update(kwargs)
    return config


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_linear_config(),
            # get_fixed_linear_config({"ROLL_IN_NUM": 2}),
            # get_fixed_linear_config({"ROLL_OUT_NUM": 2}),
            # get_fixed_linear_config({"IN_CHANNELS_DEPTH": 2}),
            # get_fixed_linear_config({"OUT_CHANNELS_PARALLELISM": 2}),
            # get_fixed_linear_config({"WEIGHT_REPEATS": 2}),
            # get_fixed_linear_config({"OUT_CHANNELS_DEPTH": 2}),
        ],
    )


def test_fixed_linear_regression():
    """
    More extensive tests to check realistic parameter sizes.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_linear_config(
                {
                    "HAS_BIAS": 1,
                    "ROLL_IN_NUM": 3 * 16 * 16,
                    "ROLL_OUT_NUM": 4,
                    "IN_CHANNELS_DEPTH": 1,
                    "OUT_CHANNELS_PARALLELISM": 4,
                    "WEIGHT_REPEATS": 14 * 14,
                    "OUT_CHANNELS_DEPTH": 96,
                }
            ),
            # get_fixed_linear_config(
            #     {
            #         "HAS_BIAS": 1,
            #         "WEIGHTS_PRE_TRANSPOSED": 0,
            #         "DATA_IN_0_TENSOR_SIZE_DIM_0": 768,
            #         "DATA_IN_0_PARALLELISM_DIM_0": 32,
            #         "WEIGHT_TENSOR_SIZE_DIM_0": 768,
            #         "WEIGHT_TENSOR_SIZE_DIM_1": 768,
            #         "WEIGHT_PARALLELISM_DIM_0": 32,
            #         "WEIGHT_PARALLELISM_DIM_1": 32,
            #         "BIAS_TENSOR_SIZE_DIM_0": 768,
            #         "BIAS_PARALLELISM_DIM_0": 32,
            #     }
            # ),
        ],
    )


if __name__ == "__main__":
    # test_fixed_linear_smoke()
    test_fixed_linear_regression()
