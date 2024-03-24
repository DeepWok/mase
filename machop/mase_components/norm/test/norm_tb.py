#!/usr/bin/env python3

import logging
from functools import partial
from random import randint
from math import ceil, log2
from copy import copy
from pathlib import Path
from os import makedirs
import pickle
# from itertools import batched  # Python 3.12

import torch
from torch import nn
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
    BatchNorm2dInteger,
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
from mase_components.norm.test.batch_norm_2d_tb import gen_mem_files

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class NormTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params([
            "TOTAL_DIM0", "TOTAL_DIM1", "COMPUTE_DIM0", "COMPUTE_DIM1",
            "DEPTH_DIM0", "DEPTH_DIM1",
            "CHANNELS", "IN_WIDTH", "IN_FRAC_WIDTH",
            "WEIGHT_WIDTH", "WEIGHT_FRAC_WIDTH",
            "OUT_WIDTH", "OUT_FRAC_WIDTH", "BATCH_NORM",
            "LAYER_NORM", "INSTANCE_NORM", "GROUP_NORM", "RMS_NORM",
            "MEM_ID",
        ])

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        # Inverse Square Root LUT
        self.isqrt_lut = make_lut(2**5, 16)

        if self.BATCH_NORM:
            self.total_channels = self.CHANNELS
            self.quantized_model = BatchNorm2dInteger(
                num_features=self.total_channels,
                affine=False,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                }
            )
        elif self.GROUP_NORM:
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
                elementwise_affine=True,
                config={
                    "data_in_width": self.IN_WIDTH,
                    "data_in_frac_width": self.IN_FRAC_WIDTH,
                    "weight_width": self.WEIGHT_WIDTH,
                    "weight_frac_width": self.WEIGHT_FRAC_WIDTH,
                }
            )
        else:
            raise Exception("Norm type is unknown.")

        self.quantized_model.eval()

        # Drivers & Monitors
        self.in_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
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

        if self.RMS_NORM:
            # Model weights (scale)
            weights = list()
            for _ in range(self.total_channels):
                weights.extend(gen_random_matrix_input(
                    *self.total_tup, *self.compute_tup, *self.in_width_tup
                ))

            # Set weight tensor into model
            weights_t = self.reconstruct_tensor(weights, self.WEIGHT_WIDTH, self.WEIGHT_FRAC_WIDTH)
            self.quantized_model.weight = nn.Parameter(weights_t)

            # Weights are same across all batches, however we need to repeat them to the driver
            repeated_weights = list()
            for _ in range(batches):
                repeated_weights.extend(weights)
            return inputs, repeated_weights

        else:
            return inputs

    def grab_mem_arr_data(self) -> tuple:
        mem_dir_arr = Path(__file__).parent / "build" / "batch_norm_2d" / "mem" / "arr"
        arr_mem_path = mem_dir_arr / f"arr_mem-{self.MEM_ID}.mem"

        arrs = {}

        with open(arr_mem_path, "rb") as f:
            arrs = pickle.load(f)

        mean = arrs["mean"]
        var = arrs["var"]
        return torch.tensor(mean), torch.tensor(var)

    def reconstruct_tensor(self, x, width, frac_width):
        batches = batched(x, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [rebuild_matrix(b, *self.total_tup, *self.compute_tup)
                       for b in batches]
        x = torch.stack(matrix_list).reshape(
            -1, self.total_channels, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, width).to(dtype=torch.float32) / (2 ** frac_width)
        return x

    def output_monitor_split(self, x, width, frac_width):
        x = integer_quantizer_for_hw(x, width, frac_width)
        x = x.reshape(-1, self.TOTAL_DIM1, self.TOTAL_DIM0)
        model_out = list()
        for i in range(x.shape[0]):
            model_out.extend(split_matrix(x[i], *self.total_tup, *self.compute_tup))
        return model_out

    def model(self, inputs):
        # Input reconstruction
        x = self.reconstruct_tensor(inputs, *self.in_width_tup)

        # Quantized Model
        if self.BATCH_NORM:
            mean, var = self.grab_mem_arr_data()
            self.quantized_model.running_mean = torch.tensor(mean)
            self.quantized_model.running_var = torch.tensor(var)
        float_y = self.quantized_model(x)

        # Output beat reconstruction
        y = self.output_monitor_split(float_y, *self.out_width_tup)

        return y

    async def run_test(self, batches=1, us=100):
        if self.RMS_NORM:
            inputs, weights = self.generate_inputs(batches=batches)
            self.in_driver.load_driver(inputs)
            self.weight_driver.load_driver(weights)
            exp_out = self.model(inputs)
            self.output_monitor.load_monitor(exp_out)
        else:
            inputs = self.generate_inputs(batches=batches)
            self.in_driver.load_driver(inputs)
            exp_out = self.model(inputs)
            self.output_monitor.load_monitor(exp_out)
        await Timer(us, 'us')
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=2, us=10)


@cocotb.test()
async def stream(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=100, us=200)


@cocotb.test()
async def backpressure(dut):
    tb = NormTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.5))
    await tb.run_test(batches=100, us=1000)


@cocotb.test()
async def valid_toggle(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    await tb.run_test(batches=100, us=1000)


@cocotb.test()
async def valid_backpressure(dut):
    tb = NormTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.5))
    await tb.run_test(batches=100, us=1000)


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
        weight_width: int = 8,
        weight_frac_width: int = 4,
        out_width: int = 8,
        out_frac_width: int = 4,
        str_id: str = "default",
        norm_type: str = "LAYER_NORM",
        mem_id: int = 0,
    ):
        lut = make_lut(2 ** LUT_POW, ISQRT_WIDTH)
        mem_path = mem_dir / f"lutmem-{str_id}.mem"
        write_memb(mem_path, lut, ISQRT_WIDTH)
        scale_mem_path, shift_mem_path = gen_mem_files(
            mem_id, in_width, in_frac_width, channels, affine=False
        )
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
            "WEIGHT_PRECISION_0": weight_width,
            "WEIGHT_PRECISION_1": weight_frac_width,
            "ISQRT_LUT_MEMFILE": verilator_str_param(str(mem_path)),
            "SCALE_LUT_MEMFILE": verilator_str_param(str(scale_mem_path)),
            "SHIFT_LUT_MEMFILE": verilator_str_param(str(shift_mem_path)),
            "MEM_ID": mem_id,
            "NORM_TYPE": verilator_str_param(norm_type.upper()),
        }
        return params

    mase_runner(
        module_param_list=[
            # Test All Supported Norm Types
            gen_cfg(norm_type="BATCH_NORM", str_id="layer"),
            gen_cfg(norm_type="LAYER_NORM", str_id="layer"),
            gen_cfg(norm_type="GROUP_NORM", str_id="group"),
            gen_cfg(norm_type="INSTANCE_NORM", str_id="inst"),
            # gen_cfg(norm_type="RMS_NORM", str_id="rms"),
        ],
        trace=True,
    )
