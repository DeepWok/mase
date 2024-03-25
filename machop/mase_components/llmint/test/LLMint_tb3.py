#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class LinearTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.in_features = dut.TENSOR_SIZE_DIM.value
        self.out_features = dut.TENSOR_SIZE_DIM.value
        self.high_slots = dut.HIGH_SLOTS.value
        self.threshold = dut.THRESHOLD.value
        self.bitwidth = dut.ORIGINAL_PRECISION.value
        self.reduced_bitwidth = dut.REDUCED_PRECISION.value
        self.weights_size = [dut.WEIGHT_DIM_0.value, dut.WEIGHT_DIM_1.value]


        print('----------------Assigned---------------')
        # print(self.in_features)
        print('features',self.in_features)
        print('type',type(self.in_features))
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_driver = StreamDriver(
            dut.clk, dut.data_in, dut.data_in_valid, dut.data_in_ready
        )

        self.weight_driver = StreamDriver(
            dut.clk, dut.weights, dut.weight_valid, dut.weight_ready
        )

        self.quantizer = partial(
            integer_quantizer, width=32, frac_width=3
        )

        self.reduced_quantizer = partial(
            integer_quantizer, width=16, frac_width=3
        )
        # Model
        self.linear_low = LinearInteger(
            in_features=4,
            out_features=4,
            bias=False,
            config={
                "data_in_width": 16,
                "data_in_frac_width": 3,
                "weight_width": 16,
                "weight_frac_width": 3,
                "bias_width": 16,
                "bias_frac_width": 3,
            },
        )

        self.linear_high= LinearInteger(
            in_features=4,
            out_features=4,
            bias=False,
            config={
                "data_in_width": 16,
                "data_in_frac_width": 3,
                "weight_width": 16,
                "weight_frac_width": 3,
                "bias_width": 16,
                "bias_frac_width": 3,
            },
        )

    print('------done----')

    def generate_inputs(self):
        return torch.randn((1, self.model.in_features))

    def preprocess_tensor(self, tensor, quantizer, config, parallelism):
        tensor = quantizer(tensor)
        tensor = (tensor * 2 ** config["frac_width"]).int()
        logger.info(f"Tensor in int format: {tensor}")
        tensor = tensor.reshape(-1, parallelism).tolist()
        return tensor

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        # Load the inputs driver
        logger.info(f"Processing inputs")
        inputs = self.preprocess_tensor(
            inputs,
            self.model.x_quantizer,
            {"widht": 16, "frac_width": 3},
            int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
        )
        self.data_in_0_driver.load_driver(inputs)

        # Load the weights driver
        logger.info(f"Processing weights")
        weights = self.preprocess_tensor(
            self.model.weight,
            self.model.w_quantizer,
            {"widht": 16, "frac_width": 3},
            int(self.dut.WEIGHT_PARALLELISM_DIM_0)
            * int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
        )
        
        self.weight_driver.load_driver(weights)

        # Load the output monitor
        logger.info(f"Processing outputs: {exp_out}")
        # To do: need to quantize output to a different precision
        outs = self.preprocess_tensor(
            exp_out,
            self.model.x_quantizer,
            {"widht": 16, "frac_width": 3},
            int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0),
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(1000, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test_20x20(dut):
    tb = LinearTB(dut)
    await tb.run_test()


if __name__ == "__main__":
    mase_runner(
        trace=True
        # module_param_list=[
        #     {
        #         "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_IN_0_PARALLELISM_DIM_0": 2,
        #         "WEIGHT_TENSOR_SIZE_DIM_0": 20,
        #         "WEIGHT_TENSOR_SIZE_DIM_1": 20,
        #         "WEIGHT_PARALLELISM_DIM_0": 20,
        #         "DATA_OUT_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_OUT_0_PARALLELISM_DIM_0": 20,
        #         "BIAS_TENSOR_SIZE_DIM_0": 20,
        #     },
        #     {
        #         "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_IN_0_PARALLELISM_DIM_0": 4,
        #         "WEIGHT_TENSOR_SIZE_DIM_0": 20,
        #         "WEIGHT_TENSOR_SIZE_DIM_1": 20,
        #         "WEIGHT_PARALLELISM_DIM_0": 20,
        #         "DATA_OUT_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_OUT_0_PARALLELISM_DIM_0": 20,
        #         "BIAS_TENSOR_SIZE_DIM_0": 20,
        #     },
        #     {
        #         "DATA_IN_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_IN_0_PARALLELISM_DIM_0": 5,
        #         "WEIGHT_TENSOR_SIZE_DIM_0": 20,
        #         "WEIGHT_TENSOR_SIZE_DIM_1": 20,
        #         "WEIGHT_PARALLELISM_DIM_0": 20,
        #         "DATA_OUT_0_TENSOR_SIZE_DIM_0": 20,
        #         "DATA_OUT_0_PARALLELISM_DIM_0": 20,
        #         "BIAS_TENSOR_SIZE_DIM_0": 20,
        #     },
        # ],
    )
