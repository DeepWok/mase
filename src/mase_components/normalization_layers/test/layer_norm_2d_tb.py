#!/usr/bin/env python3

import os
import pytest

import torch
import logging
from functools import partial
from src.mase_components.helper import generate_memory
from pathlib import Path
import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge
import logging

logger = logging.getLogger("norm.models")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import fixed_preprocess_tensor

from mase_cocotb.utils import bit_driver
from chop.nn.quantizers import integer_floor_quantizer
from chop.nn.quantized.modules import LayerNormIntegerFloor
def quantize(x, width, frac_width, by_pass=False):
    if not by_pass:
        x = integer_floor_quantizer(x, width, frac_width)
    return x
def get_dim_and_prodofdim(x, normalized_shape):
    dim = tuple(range(0 - len(normalized_shape), 0))
    num_vals = 1
    for items in dim:
        num_vals *= x.shape[items]
    return dim, num_vals
def isqrt(x:torch.Tensor):
    x = (x+1e-5).sqrt()
    x = x.reciprocal()
    return x
from math import ceil, log2
def _fixed_group_norm_2d_model(
    x: torch.Tensor,
    normalized_shape: tuple or int,
    q_config,
    floor=True,
):
    #TODO: add hardware debug info
    #TODO: support floor
    logger.debug(f"Input: \n {x}")
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    dim, num_vals = get_dim_and_prodofdim(x, normalized_shape)
    x = quantize(x, q_config["data_in_width"], q_config["data_in_frac_width"], q_config["by_pass"])
    logger.debug(f"Input Quantized: \n {x}")
    num_vals_frac_width = ceil(log2(num_vals))
    inv_num_vals_quant = quantize(torch.tensor(1/num_vals),num_vals_frac_width + 2, num_vals_frac_width)
    logger.debug(f"Input Quantized: \n {inv_num_vals_quant}")
    # Mean calculation
    mu_acc = x.sum(dim, keepdim=True)
    logger.debug(f"Mu Acc: \n {mu_acc}")
    mu = mu_acc * inv_num_vals_quant
    logger.debug(f"Mu: \n {mu}")
    mu = quantize(mu, q_config["data_in_width"], q_config["data_in_frac_width"], q_config["by_pass"])
    logger.debug(f"Mu Quantized: \n {mu}")
    #I hope the output precision here should be $clog2
    # Variance calculation
    diff = x - mu
    logger.debug(f"Diff: \n {diff}")

    squares = diff**2
    logger.debug(f"Squares: {squares}")

    sum_squares = torch.sum(squares, dim, keepdim=True)

    var = sum_squares * inv_num_vals_quant
    var = quantize(var, q_config["isqrt_in_width"], q_config["isqrt_in_frac_width"], q_config["by_pass"])
    logger.debug(f"Variance: \n {var}")

    inv_sqrt = isqrt(var)
    inv_sqrt = quantize(inv_sqrt, q_config["isqrt_out_width"], q_config["isqrt_out_frac_width"], q_config["by_pass"])
    logger.debug(f"INV SQRT INT: \n {inv_sqrt}")

    # Norm calculation
    norm_out = diff * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out)

    norm_out = quantize(norm_out, q_config["data_out_width"], q_config["data_out_frac_width"], q_config["by_pass"])
    logger.debug(f"Norm (Casted): \n {norm_out}")

    return norm_out 
class LayerNormTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.in_data_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        self.out_data_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        # Model
        self.model = LayerNormIntegerFloor(
            normalized_shape=self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "isqrt_in_width": self.get_parameter("ISQRT_IN_PRECISION_0"),
                "isqrt_in_frac_width": self.get_parameter("ISQRT_IN_PRECISION_1"),
                "isqrt_out_width": self.get_parameter("ISQRT_OUT_PRECISION_0"),
                "isqrt_out_frac_width": self.get_parameter("ISQRT_OUT_PRECISION_1"),
                "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "data_out_frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                "by_pass": False,
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        self.in_data_driver.log.setLevel(logging.DEBUG)
        self.out_data_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        return torch.randn(
            (
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, batches, us):
        await self.reset()
        self.log.info(f"Reset finished")

        for _ in range(batches):
            inputs = self.generate_inputs()
            exp_out = self.model(inputs)

            # * Load the inputs driver
            self.log.info(f"Processing inputs: {inputs}")
            inputs = fixed_preprocess_tensor(
                tensor=inputs,
                q_config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
                ],
                floor=True,
            )
            self.in_data_driver.load_driver(inputs)

            # * Load the output monitor
            self.log.info(f"Processing outputs: {exp_out}")
            outs = fixed_preprocess_tensor(
                tensor=exp_out,
                q_config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
                ],
            )
            self.out_data_monitor.load_monitor(outs)
            cocotb.start_soon(check_signal(self.dut, self.log))

        await Timer(us, units="us")
        assert self.out_data_monitor.exp_queue.empty()

async def check_signal(dut, log):
    while True:
        await RisingEdge(dut.clk)
        handshake_signal_check(
            dut.mu_acc_valid, 
            dut.mu_acc_ready, 
            dut.mu_acc, log,
            dut.DATA_IN_0_PRECISION_1.value)
        handshake_signal_check(
            dut.mu_acc_valid, 
            dut.mu_acc_ready, 
            dut.mu_in, log,
            dut.DATA_IN_0_PRECISION_0.value)
        # breakpoint()
        a = 1
        # handshake_signal_check(dut.rolled_k_valid, dut.rolled_k_ready, dut.rolled_k, log)
        # handshake_signal_check(dut.bias_valid,
        #                        dut.bias_ready,
        #                        dut.bias, log)


def handshake_signal_check(valid, ready, signal, log, frac_width=0):
    scale = 2**frac_width
    if isinstance(signal.value, list):
        svalue = [i.signed_integer / scale for i in signal.value]
    else:
        svalue=signal.value.signed_integer / scale
    if valid.value & ready.value:
        log.debug(f"handshake {str(signal)} = {svalue}")

@cocotb.test()
async def single_test(dut):
    tb = LayerNormTB(dut)
    tb.out_data_monitor.ready.value = 1
    await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = LayerNormTB(dut)
#     tb.out_data_monitor.ready.value = 1
#     await tb.run_test(batches=100, us=2000)


@cocotb.test()
async def repeated_mult_backpressure(dut):
    tb = LayerNormTB(dut)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
    await tb.run_test(batches=10, us=500)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = LayerNormTB(dut)
    tb.in_data_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
    await tb.run_test(batches=50, us=200)

# Don't support :
# 1. DATA_IN_0_PARALLELISM_DIM_0 ==DATA_IN_0_TENSOR_SIZE_DIM_0
# 
dut_params = {
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 256,
    "DATA_IN_0_PARALLELISM_DIM_0": 4,
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "ISQRT_IN_PRECISION_0": 7,
    "ISQRT_IN_PRECISION_1": 4,
    "ISQRT_OUT_PRECISION_0": 12,
    "ISQRT_OUT_PRECISION_1": 4,
    "DATA_OUT_0_PRECISION_0": 10,
    "DATA_OUT_0_PRECISION_1": 4,
    "GROUP_CHANNELS":1,
}


def get_fixed_softmax_config(kwargs={}):
    config = dut_params
    config.update(kwargs)
    return config


torch.manual_seed(1)
@pytest.mark.dev
def test_fixed_softmax_smoke():
    """
    Some quick tests to check if the module is working.
    """
    path = Path(__file__).parents[1] / "rtl"
    generate_memory.generate_sv_lut(
        "isqrt",
        dut_params["ISQRT_IN_PRECISION_0"],
        dut_params["ISQRT_IN_PRECISION_1"],
        dut_params["ISQRT_OUT_PRECISION_0"],
        dut_params["ISQRT_OUT_PRECISION_1"],
        path=path,
        floor=True,
    )
    mase_runner(
        trace=True,
        module_param_list=[
            get_fixed_softmax_config(),
        ],
        # skip_build=True,
    )


if __name__ == "__main__":
    test_fixed_softmax_smoke()
