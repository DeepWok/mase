#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
import pdb
import cocotb
from functools import partial
from cocotb.triggers import *
from chop.passes.graph.transforms.quantize.quantizers import *
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t
from torch.autograd.function import InplaceFunction

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class hardswish(torch.nn.Hardswish):
    def forward(x):
        return 3.0 * (x * (x + 3.0)) / 16.0


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply


# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class HardswishTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        # pdb.set_trace()

        self.assign_self_params(
            [
                "DATA_IN_0_PRECISION_0",
                "DATA_IN_0_PRECISION_1",
                "DATA_IN_0_TENSOR_SIZE_DIM_0",
                "DATA_IN_0_TENSOR_SIZE_DIM_1",
                "DATA_IN_0_PARALLELISM_DIM_0",
                "DATA_IN_0_PARALLELISM_DIM_1",
                "DATA_OUT_0_PRECISION_0",
                "DATA_OUT_0_PRECISION_1",
                "DATA_OUT_0_TENSOR_SIZE_DIM_0",
                "DATA_OUT_0_TENSOR_SIZE_DIM_1",
                "DATA_OUT_0_PARALLELISM_DIM_0",
                "DATA_OUT_0_PARALLELISM_DIM_1",
            ]
        )
        # pdb.set_trace()
        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        # self.weight_driver = StreamDriver(
        #     dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        # )
        # self.bias_driver = StreamDriver(
        #     dut.clk, dut.data_in_0, dut.bias_valid, dut.bias_ready
        # )
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )
        self.linear_layer = torch.nn.Hardswish()

    def exp(self, inputs):
        # Run the model with the provided inputs and return the outputs
        reals = torch.nn.functional.leaky_relu(inputs)
        qreals = self.dquantizer(reals)
        qints = qreals * 2**self.fracw
        qints = qints.to(torch.int64)
        qclamp = qints.clamp(
            min=-(2 ** (self.w - self.fracw - 1)),
            max=2 ** (self.w - self.fracw - 1) - 1,
        )

    def generate_inputs(self, w, fracw):
        self.dquantizer = partial(integer_quantizer, width=w, frac_width=fracw)
        self.w = w
        self.fracw = fracw
        realinp = torch.randn(10)
        inputs = self.dquantizer(realinp)
        intinp = (inputs * 2**self.fracw).to(torch.int64)
        intinp.clamp(
            min=-(2 ** (self.w - fracw - 1)), max=2 ** (self.w - fracw - 1) - 1
        )
        print(intinp)
        pdb.set_trace()
        return intinp


def floattofixed(x, dw, fracw):
    out = (x * 2.0**fracw).to(torch.int32)
    return out


@cocotb.test()
async def test(dut):
    # pdb.set_trace()
    tb = HardswishTB(dut)
    await tb.reset()
    logger.info(f"Reset finished")
    tb.data_out_0_monitor.ready.value = 1

    inputs = tb.generate_inputs(8, 4)
    # logger.info(f"inputs: {inputs}, q_inputs: {q_inputs}")
    exp_out = tb.exp(inputs)

    # To do: replace with tb.load_drivers(inputs)
    tb.data_in_0_driver.append(inputs.tolist())
    # tb.weight_driver.append(
    #     tb.linear_layer.w_quantizer(tb.linear_layer.weight).tolist()
    # )
    # tb.bias_driver.append(self.linear_layer.b_quantizer(self.linear_layer.bias))

    # To do: replace with tb.load_monitors(exp_out)
    tb.data_out_0_monitor.expect(exp_out)
    # To do: replace with tb.run()
    # pdb.set_trace()
    await Timer(1000, units="us")

    # To do: replace with tb.monitors_done() --> for monitor, call monitor_done()
    assert tb.data_out_0_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(
        module_param_list=[
            {
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 10,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 10,
                "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 10,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
            }
        ]
    )
