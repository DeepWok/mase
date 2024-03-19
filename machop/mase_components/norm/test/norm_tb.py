#!/usr/bin/env python3

import logging
from functools import partial
from random import randint
from math import ceil, log2
from copy import copy
from pathlib import Path
from os import makedirs
# from itertools import batched  # Python 3.12

import torch
import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)
from mase_cocotb.utils import (
    bit_driver,
    batched,
    sign_extend_t,
    verilator_str_param,
)

from chop.passes.graph.transforms.quantize.quantized_modules import (
    LayerNormInteger,
    GroupNormInteger,
    InstanceNorm2dInteger,
)
from chop.passes.graph.transforms.quantize.quantized_modules.rms_norm import (
    RMSNormInteger
)
from chop.passes.graph.transforms.quantize.quantizers.quantizers_for_hw import (
    integer_quantizer_for_hw
)
from mase_components.fixed_arithmetic.test.isqrt_sw import make_lut
from mase_components.common.test.lut_tb import write_memb

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class NormTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "DEPTH_DIM0", "DEPTH_DIM1",
            "CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "LAYER_NORM", "INSTANCE_NORM", "GROUP_NORM", "RMS_NORM",
        ])

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Inverse Square Root LUT
        self.isqrt_lut = make_lut(2**5, 16)

        if self.GROUP_NORM:
            self.num_groups = randint(2, 3)  # Random number of groups
            self.total_channels = self.CHANNELS * self.num_groups
            self.quantized_model = GroupNormInteger(
                num_groups=self.num_groups,
                num_channels=self.total_channels,
                affine=False,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                }
            )
        elif self.LAYER_NORM:
            self.total_channels = self.CHANNELS
            self.quantized_model = LayerNormInteger(
                normalized_shape=[self.total_channels, self.TOTAL_DIM1, self.TOTAL_DIM0],
                elementwise_affine=False,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                }
            )
        elif self.INSTANCE_NORM:
            self.total_channels = randint(3, 4)  # Random number of total channels
            self.quantized_model = InstanceNorm2dInteger(
                num_features=self.total_channels,
                affine=False,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                }
            )
        elif self.RMS_NORM:
            self.total_channels = self.CHANNELS
            self.quantized_model = RMSNormInteger(
                normalized_shape=[self.total_channels, self.TOTAL_DIM1, self.TOTAL_DIM0],
                elementwise_affine=False,
                bias=False,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                }
            )
        else:
            raise Exception("Norm type is unknown.")

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        # Bit Error calculation
        error_bits = 2

        # If we want the output frac to have larger width, we can expect a
        # larger rounding error difference between the integer and float models
        if self.OUT_FRAC_WIDTH > self.IN_FRAC_WIDTH:
            error_bits += 2 ** (self.OUT_FRAC_WIDTH - self.IN_FRAC_WIDTH)

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready,
            name="Output Monitor",
            width=self.OUT_WIDTH,
            signed=True,
            error_bits=error_bits,
        )

    def generate_inputs(self, batches=1):
        inputs = list()
        for _ in range(self.total_channels * batches):
            inputs.extend(gen_random_matrix_input(
                *self.total_tup, *self.compute_tup, *self.in_width_tup
            ))
        return inputs

    def assert_all_monitors_empty(self):
        assert self.output_monitor.exp_queue.empty()

    def model(self, inputs):
        # Input reconstruction
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list).reshape(
            -1, self.total_channels, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (2 ** self.IN_FRAC_WIDTH)

        # Float Model
        float_y = self.quantized_model(x)

        # Output beat reconstruction
        y = integer_quantizer_for_hw(float_y, self.OUT_WIDTH, self.OUT_FRAC_WIDTH)
        y = y.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(y.shape[0]):
            model_out.extend(split_matrix(y[i], *self.total_tup, *self.compute_tup))

        return model_out

    def setup_test(self, batches=1):
        inputs = self.generate_inputs(batches=batches)
        self.in_driver.load_driver(inputs)
        exp_out = self.model(inputs)
        self.output_monitor.load_monitor(exp_out)


@cocotb.test()
async def basic(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=2)
    await Timer(10, 'us')
    tb.assert_all_monitors_empty()


@cocotb.test()
async def stream(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=100)
    await Timer(200, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def backpressure(dut):
    tb = NormTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.5))
    tb.setup_test(batches=200)
    await Timer(400, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_toggle(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    tb.setup_test(batches=200)
    await Timer(400, 'us')
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_backpressure(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.5))
    tb.setup_test(batches=200)

    await Timer(400, 'us')
    assert tb.output_monitor.exp_queue.empty()


if __name__ == "__main__":
    # Consts
    LUT_POW = 5
    ISQRT_WIDTH = 16

    mem_dir = Path(__file__).parent / "build" / "norm" / "mem"
    makedirs(mem_dir, exist_ok=True)

    def gen_cfg(
        total_dim0: int = 4,
        total_dim1: int = 4,
        compute_dim0: int = 2,
        compute_dim1: int = 2,
        channels: int = 2,
        in_width: int = 8,
        in_frac_width: int = 4,
        out_width: int = 8,
        out_frac_width: int = 4,
        str_id: str = "default",
        norm_type: str = "LAYER_NORM",
    ):
        lut = make_lut(2 ** LUT_POW, ISQRT_WIDTH)
        mem_path = mem_dir / f"lutmem-{str_id}.mem"
        write_memb(mem_path, lut, ISQRT_WIDTH)
        params = {
            "DATA_IN_0_TENSOR_SIZE_DIM_0": total_dim0,
            "DATA_IN_0_TENSOR_SIZE_DIM_1": total_dim1,
            "DATA_IN_0_PARALLELISM_DIM_0": compute_dim0,
            "DATA_IN_0_PARALLELISM_DIM_1": compute_dim1,
            "DATA_IN_0_PARALLELISM_DIM_2": channels,
            "DATA_IN_0_PRECISION_0": in_width,
            "DATA_IN_0_PRECISION_1": in_frac_width,
            "DATA_OUT_0_PRECISION_0": out_width,
            "DATA_OUT_0_PRECISION_1": out_frac_width,
            "ISQRT_LUT_MEMFILE": verilator_str_param(str(mem_path)),
            "NORM_TYPE": verilator_str_param(norm_type.upper()),
        }
        return params

    mase_runner(
        module_param_list=[
            # Test All Supported Norm Types
            gen_cfg(norm_type="LAYER_NORM", str_id="layer"),
            gen_cfg(norm_type="GROUP_NORM", str_id="group"),
            gen_cfg(norm_type="INSTANCE_NORM", str_id="inst"),
            gen_cfg(norm_type="RMS_NORM", str_id="rms"),
        ],
    )
