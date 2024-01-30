#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class LinearTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "DATA_IN_0_PRECISION_0",
                "DATA_IN_0_PRECISION_1",
                "DATA_IN_0_TENSOR_SIZE_DIM_0",
                "DATA_IN_0_TENSOR_SIZE_DIM_1",
                "DATA_IN_0_PARALLELISM_DIM_0",
                "DATA_IN_0_PARALLELISM_DIM_1",
                "IN_0_DEPTH",
                "WEIGHT_PRECISION_0",
                "WEIGHT_PRECISION_1",
                "WEIGHT_TENSOR_SIZE_DIM_0",
                "WEIGHT_TENSOR_SIZE_DIM_1",
                "WEIGHT_PARALLELISM_DIM_0",
                "WEIGHT_PARALLELISM_DIM_1",
                "DATA_OUT_0_PRECISION_0",
                "DATA_OUT_0_PRECISION_1",
                "DATA_OUT_0_TENSOR_SIZE_DIM_0",
                "DATA_OUT_0_TENSOR_SIZE_DIM_1",
                "DATA_OUT_0_PARALLELISM_DIM_0",
                "DATA_OUT_0_PARALLELISM_DIM_1",
                "BIAS_PRECISION_0",
                "BIAS_PRECISION_1",
                "BIAS_TENSOR_SIZE_DIM_0",
                "BIAS_TENSOR_SIZE_DIM_1",
                "BIAS_PARALLELISM_DIM_0",
                "BIAS_PARALLELISM_DIM_1",
            ]
        )

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )
        # self.bias_driver = StreamDriver(
        #     dut.clk, dut.data_in_0, dut.bias_valid, dut.bias_ready
        # )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

        # Model
        # self.linear_layer = LinearInteger(
        #     in_features=784,
        #     out_features=10,
        #     bias=False,
        #     config={
        #         "data_in_width": 16,
        #         "data_in_frac_width": 3,
        #         "weight_width": 16,
        #         "weight_frac_width": 3,
        #         "bias_width": 16,
        #         "bias_frac_width": 3,
        #     },
        # )
        self.linear_layer = None

    def generate_inputs(self):
        return torch.randn((1, 784))

    def model(self, inputs):
        # Run the model with the provided inputs and return the outputs
        out = self.linear_layer(inputs)
        return out


@cocotb.test()
async def test(dut):
    tb = LinearTB(dut)
    await tb.reset()
    logger.info(f"Reset finished")
    tb.data_out_0_monitor.ready.value = 1

    inputs = tb.generate_inputs()
    # logger.info(f"inputs: {inputs}, q_inputs: {q_inputs}")
    exp_out = tb.model(inputs)

    # To do: replace with tb.load_drivers(inputs)
    tb.data_in_0_driver.append(inputs.tolist())
    tb.weight_driver.append(
        tb.linear_layer.w_quantizer(tb.linear_layer.weight).tolist()
    )
    # tb.bias_driver.append(self.linear_layer.b_quantizer(self.linear_layer.bias))

    # To do: replace with tb.load_monitors(exp_out)
    tb.data_out_0_monitor.expect(exp_out)

    # To do: replace with tb.run()
    await Timer(100, units="us")
    # To do: replace with tb.monitors_done() --> for monitor, call monitor_done()
    assert tb.data_out_0_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(
        module_param_list=[
            {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 4,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "WEIGHT_TENSOR_SIZE_DIM_0": 32,
                "WEIGHT_TENSOR_SIZE_DIM_1": 1,
                "WEIGHT_PARALLELISM_DIM_0": 4,
                "WEIGHT_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 4,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
                "BIAS_TENSOR_SIZE_DIM_0": 4,
                "BIAS_TENSOR_SIZE_DIM_1": 1,
                "BIAS_PARALLELISM_DIM_0": 4,
                "BIAS_PARALLELISM_DIM_1": 1,
            }
        ]
    )
