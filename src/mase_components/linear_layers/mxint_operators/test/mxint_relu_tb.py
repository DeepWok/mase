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
from utils import MXIntRelu


class MXIntReluTB(Testbench):
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

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )

        # Model
        self.model = MXIntRelu(
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "data_in_parallelism_dim_1": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_1"
                ),
                "data_in_parallelism_dim_0": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ),
            },
            bypass=True,
        )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        self.log.info(f"Mantissa Tensor: {mtensor}")
        self.log.info(f"Exponenr Tensor: {etensor}")
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return qtensor, tensor_inputs

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

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
        quantized, inputs = self.preprocess_tensor_for_mxint(
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

        exp_out = self.model(quantized)
        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        _, outs = self.preprocess_tensor_for_mxint(
            tensor=exp_out,
            config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = MXIntReluTB(dut)
    await tb.run_test(us=100)


def get_relu_config(kwargs={}):
    config = {
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 2,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 1,
    }

    config.update(kwargs)
    return config


@pytest.mark.dev
def test_relu():
    """
    More extensive tests to check realistic parameter sizes.
    """
    mase_runner(
        trace=True,
        module_param_list=[
            get_relu_config(
                {
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
                    "DATA_IN_0_PARALLELISM_DIM_0": 2,
                }
            ),
        ],
    )


if __name__ == "__main__":
    test_relu()
