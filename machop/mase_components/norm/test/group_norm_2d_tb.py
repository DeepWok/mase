#!/usr/bin/env python3

import logging
from random import randint
# from itertools import batched  # Python 3.12

import torch
import torch.nn.functional as F
import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_tools import gen_random_matrix_input, rebuild_matrix, split_matrix
from mase_cocotb.utils import bit_driver

from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer, integer_quantizer_for_hw


logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

# Apparently this function only exists in Python 3.12 ...
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class GroupNorm2dTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "GROUP_CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "DEPTH_DIM0", "DEPTH_DIM1"
        ])

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.in_data, dut.in_valid, dut.in_ready
        )
        self.output_monitor = StreamMonitor(
            dut.clk, dut.out_data, dut.out_valid, dut.out_ready, check=True
        )

    def generate_inputs(self):
        inputs = list()
        for _ in range(self.GROUP_CHANNELS):
            inputs.extend(gen_random_matrix_input(
                *self.total_tup, *self.compute_tup, *self.in_width_tup
            ))
        return inputs

    def model(self, inputs):
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list)
        y = F.layer_norm(
            x.to(dtype=torch.float32), [self.TOTAL_DIM1, self.TOTAL_DIM0]
        )
        # y_int = integer_quantizer(y, *self.out_width_tup)
        # print(y_int)
        y_int_hw = integer_quantizer_for_hw(y, *self.out_width_tup)
        # print(y_int_hw)
        model_out = list()
        for i in range(y_int_hw.shape[0]):
            model_out.extend(
                split_matrix(y_int_hw[i], *self.total_tup, *self.compute_tup)
            )
        return model_out


@cocotb.test()
async def basic(dut):
    tb = GroupNorm2dTB(dut)
    await tb.reset()

    inputs = tb.generate_inputs()
    tb.in_driver.load_driver(inputs)
    exp_out = tb.model(inputs)
    tb.output_monitor.load_monitor(exp_out)

    await Timer(100, 'us')
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(trace=True)
