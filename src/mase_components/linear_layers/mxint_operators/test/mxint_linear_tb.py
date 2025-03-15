#!/usr/bin/env python3

import os, pytest

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)
from mase_cocotb.runner import mase_runner

torch.manual_seed(0)
# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from utils import MXIntLinear


class LinearTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready,
        )
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mweight, dut.eweight), dut.weight_valid, dut.weight_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_driver = MultiSignalStreamDriver(
                dut.clk, (dut.mbias, dut.ebias), dut.bias_valid, dut.bias_ready
            )
            self.bias_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

        # Model
        self.model = MXIntLinear(
            in_features=self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            out_features=self.get_parameter("DATA_OUT_0_TENSOR_SIZE_DIM_0"),
            bias=True if self.get_parameter("HAS_BIAS") == 1 else False,
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "data_in_parallelism_dim_1": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_1"
                ),
                "data_in_parallelism_dim_0": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ),
                "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "weight_exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "weight_parallelism_dim_1": self.get_parameter(
                    "WEIGHT_PARALLELISM_DIM_1"
                ),
                "weight_parallelism_dim_0": self.get_parameter(
                    "WEIGHT_PARALLELISM_DIM_0"
                ),
                "bias_width": self.get_parameter("BIAS_PRECISION_0"),
                "bias_exponent_width": self.get_parameter("BIAS_PRECISION_1"),
                "bias_parallelism_dim_1": self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                "bias_parallelism_dim_0": self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "data_out_exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                "data_out_parallelism_dim_1": self.get_parameter(
                    "DATA_OUT_0_PARALLELISM_DIM_1"
                ),
                "data_out_parallelism_dim_0": self.get_parameter(
                    "DATA_OUT_0_PARALLELISM_DIM_0"
                ),
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        self.log.info(f"Mantissa Tensor: {mtensor}")
        self.log.info(f"Exponenr Tensor: {etensor}")
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

    def generate_inputs(self):
        return torch.randn(
            (
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, us):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
        inputs = self.preprocess_tensor_for_mxint(
            tensor=inputs,
            config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_in_0_driver.load_driver(inputs)

        # * Load the weights driver
        weights = self.model.weight

        self.log.info(f"Processing weights: {weights}")
        weights = self.preprocess_tensor_for_mxint(
            tensor=weights,
            config={
                "width": self.get_parameter("WEIGHT_PRECISION_0"),
                "exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
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
            bias = self.preprocess_tensor_for_mxint(
                tensor=bias,
                config={
                    "width": self.get_parameter("BIAS_PRECISION_0"),
                    "exponent_width": self.get_parameter("BIAS_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                    self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                ],
            )
            self.bias_driver.load_driver(bias)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        outs = self.preprocess_tensor_for_mxint(
            tensor=exp_out,
            config={
                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = LinearTB(dut)
    await tb.run_test(us=100)


def get_fixed_linear_config(kwargs={}):
    # if pretranspose
    #   weight1 = in0
    # else
    #   weight0 = in0
    # currently, we only consider the transposed situation
    config = {
        "HAS_BIAS": 0,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 2,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": 2,
        "WEIGHT_TENSOR_SIZE_DIM_1": 2,
        "WEIGHT_PARALLELISM_DIM_0": 2,
        "WEIGHT_PARALLELISM_DIM_1": 1,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 4,
        "WEIGHT_PRECISION_0": 8,
        "WEIGHT_PRECISION_1": 4,
        "BIAS_PRECISION_0": 8,
        "BIAS_PRECISION_1": 4,
        "DATA_OUT_0_PRECISION_0": 10,
        "DATA_OUT_0_PRECISION_1": 4,
    }
    # config = {
    #     "HAS_BIAS": 1,
    #     "DATA_IN_0_TENSOR_SIZE_DIM_0": 32,
    #     "DATA_IN_0_TENSOR_SIZE_DIM_1": 16,
    #     "DATA_IN_0_PARALLELISM_DIM_0": 4,
    #     "DATA_IN_0_PARALLELISM_DIM_1": 4,
    #     "WEIGHT_TENSOR_SIZE_DIM_0": 32,
    #     "WEIGHT_TENSOR_SIZE_DIM_1": 16,
    #     "WEIGHT_PARALLELISM_DIM_0": 4,
    #     "WEIGHT_PARALLELISM_DIM_1": 4,
    #     "DATA_IN_0_PRECISION_0": 9,
    #     "DATA_IN_0_PRECISION_1": 4,
    #     "WEIGHT_PRECISION_0": 8,
    #     "WEIGHT_PRECISION_1": 3,
    #     "BIAS_PRECISION_0": 8,
    #     "BIAS_PRECISION_1": 4,
    #     "DATA_OUT_0_PRECISION_0": 12,
    #     "DATA_OUT_0_PRECISION_1": 4,
    # }
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
            # noticed here if change WEIGHT_PRE_TRANSPOSED also need to change the DIM_SIZE to match ACTIVATION
            # get_fixed_linear_config(
            #     {
            #         "WEIGHT_TENSOR_SIZE_DIM_0": 32,
            #         "WEIGHT_TENSOR_SIZE_DIM_1": 16,
            #         "WEIGHT_PARALLELISM_DIM_0": 4,
            #         "WEIGHT_PARALLELISM_DIM_1": 2,
            #     },
            # ),
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
                    "HAS_BIAS": 0,
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
                    "DATA_IN_0_PARALLELISM_DIM_0": 2,
                    "WEIGHT_TENSOR_SIZE_DIM_0": 4,
                    "WEIGHT_TENSOR_SIZE_DIM_1": 4,
                    "WEIGHT_PARALLELISM_DIM_0": 2,
                    "WEIGHT_PARALLELISM_DIM_1": 2,
                    "BIAS_TENSOR_SIZE_DIM_0": 4,
                    "BIAS_PARALLELISM_DIM_0": 2,
                }
            ),
            # get_fixed_linear_config(
            #     {
            #         "HAS_BIAS": 1,
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
