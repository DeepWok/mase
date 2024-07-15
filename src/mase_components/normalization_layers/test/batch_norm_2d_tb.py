#!/usr/bin/env python3

import logging
from functools import partial
from random import randint, uniform
from math import ceil, log2, sqrt
from copy import copy
from pathlib import Path
from os import makedirs

# from itertools import batched  # Python 3.12

import torch
from torch import Tensor
import numpy as np
import cocotb
from cocotb.result import TestFailure
from cocotb.triggers import *
import pickle

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

from chop.nn.quantized.modules import BatchNorm2dInteger
from chop.nn.quantizers import integer_floor_quantizer, integer_quantizer
from chop.nn.quantizers.quantizers_for_hw import (
    integer_floor_quantizer_for_hw,
    integer_quantizer_for_hw,
)


from mase_components.cast.test.fixed_signed_cast_tb import _fixed_signed_cast_model
from mase_components.scalar_operators.fixed.test.isqrt_sw import lut_parameter_dict
from mase_components.common.test.lut_tb import write_memb, read_memb

import pytest

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class BatchNorm2dTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "TOTAL_DIM0",
                "TOTAL_DIM1",
                "COMPUTE_DIM0",
                "COMPUTE_DIM1",
                "NUM_CHANNELS",
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                # Local parameters.
                "DEPTH_DIM0",
                "DEPTH_DIM1",
                "MEM_ID",
                "AFFINE",
            ]
        )

        # Helper tuples
        self.total_tup = self.TOTAL_DIM0, self.TOTAL_DIM1
        self.compute_tup = self.COMPUTE_DIM0, self.COMPUTE_DIM1
        self.in_width_tup = self.IN_WIDTH, self.IN_FRAC_WIDTH
        self.out_width_tup = self.OUT_WIDTH, self.OUT_FRAC_WIDTH

        self.quantized_model = BatchNorm2dInteger(
            num_features=self.NUM_CHANNELS,
            affine=False,
            config={
                "data_in_width": self.IN_WIDTH,
                "data_in_frac_width": self.IN_FRAC_WIDTH,
            },
        )

        # Drivers & Monitors
        self.in_driver = StreamDriver(dut.clk, dut.in_data, dut.in_valid, dut.in_ready)
        # Error calculation
        error_bits = 2

        # If we want the output frac to have larger width, we can expect a
        # larger rounding error difference between the integer and float models
        if self.OUT_FRAC_WIDTH > self.IN_FRAC_WIDTH:
            error_bits += 2 ** (self.OUT_FRAC_WIDTH - self.IN_FRAC_WIDTH)

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_data,
            dut.out_valid,
            dut.out_ready,
            name="Output Monitor",
            width=self.OUT_WIDTH,
            signed=True,
            error_bits=error_bits,
        )
        # self.output_monitor.log.setLevel("DEBUG")

    def generate_inputs(self, batches=1):
        inputs = list()
        for _ in range(self.NUM_CHANNELS * batches):
            inputs.extend(
                gen_random_matrix_input(
                    *self.total_tup, *self.compute_tup, *self.in_width_tup
                )
            )

        return inputs

    def assert_all_monitors_empty(self):
        assert self.output_monitor.exp_queue.empty()

    def grab_mem_arr_data(self) -> tuple:
        mem_dir_arr = Path(__file__).parent / "build" / "batch_norm_2d" / "mem" / "arr"
        arr_mem_path = mem_dir_arr / f"arr_mem-{self.MEM_ID}.mem"

        arrs = {}

        with open(arr_mem_path, "rb") as f:
            arrs = pickle.load(f)

        mean = arrs["mean"]
        var = arrs["var"]
        gamma = arrs["gamma"] if self.AFFINE else None
        beta = arrs["beta"] if self.AFFINE else None

        if self.AFFINE:
            return (
                torch.tensor(mean),
                torch.tensor(var),
                torch.tensor(gamma),
                torch.tensor(beta),
            )
        else:
            return torch.tensor(mean), torch.tensor(var)

    def model(self, inputs):
        # Input reconstruction
        batches = batched(inputs, self.DEPTH_DIM0 * self.DEPTH_DIM1)
        matrix_list = [
            rebuild_matrix(b, *self.total_tup, *self.compute_tup) for b in batches
        ]
        x = torch.stack(matrix_list).reshape(
            -1, self.NUM_CHANNELS, self.TOTAL_DIM1, self.TOTAL_DIM0
        )
        x = sign_extend_t(x, self.IN_WIDTH).to(dtype=torch.float32) / (
            2**self.IN_FRAC_WIDTH
        )

        # Float Model
        self.quantized_model.training = False
        mean, var = self.grab_mem_arr_data()
        self.quantized_model.running_mean = torch.tensor(mean)
        self.quantized_model.running_var = torch.tensor(var)
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


def integer_quantizer_list(x: list, width: int, frac_width: int) -> list:
    t = torch.tensor(x)
    return integer_quantizer(t, width, frac_width).numpy().tolist()


def integer_quantizer_list_hw(x: list, width: int, frac_width: int) -> list:
    t = torch.tensor(x)
    return integer_quantizer_for_hw(t, width, frac_width).numpy().tolist()


def write_float_mem(x: dict, filepath: Path) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(x, f)


def gen_mem_files(mem_id, width, frac_width, channels, affine=False):
    # Make the directories
    mem_dir = Path(__file__).parent / "build" / "batch_norm_2d" / "mem"
    makedirs(mem_dir, exist_ok=True)
    mem_dir_arr = Path(__file__).parent / "build" / "batch_norm_2d" / "mem" / "arr"
    makedirs(mem_dir_arr, exist_ok=True)

    # Generate mean, var, gamma and beta vectors
    max_number = (2 ** (width - 1) - 1) / 2 ** (frac_width)
    min_number = -(2 ** (width - 1))
    min_number_var = 2 ** (frac_width)
    mean_f = [max_number * uniform(min_number, max_number) for _ in range(channels)]
    var_f = [
        max_number * uniform(min_number_var, max_number) for _ in range(1, channels + 1)
    ]  # NOTE: variance cannot be negative or 0

    if affine:
        gamma_f = [
            max_number * uniform(min_number, max_number) for _ in range(channels)
        ]
        beta_f = [max_number * uniform(min_number, max_number) for _ in range(channels)]

    # Quantized lists
    mean = integer_quantizer_list(mean_f, width, frac_width)
    var = integer_quantizer_list(var_f, width, frac_width)
    if affine:
        gamma = integer_quantizer_list(gamma_f, width, frac_width)
        beta = integer_quantizer_list(beta_f, width, frac_width)

    # Calculate scale and shift LUTs
    if affine:
        # Generate floating point representation
        scale = [g / sqrt(variance) for variance, g in zip(var, gamma)]
        shift = [
            b - mu * g / sqrt(variance)
            for mu, variance, g, b in zip(mean, var, gamma, beta)
        ]
    else:
        # Generate floating point representation
        scale = [1 / sqrt(variance) for variance in var]
        shift = [-mu / sqrt(variance) for mu, variance in zip(mean, var)]

    var = [1 / (sca) ** 2 for sca in integer_quantizer_list(scale, width, frac_width)]
    var = integer_quantizer_list(var, width, frac_width)

    mean = [
        -mu / sca
        for sca, mu in zip(
            integer_quantizer_list(scale, width, frac_width),
            integer_quantizer_list(shift, width, frac_width),
        )
    ]
    mean = integer_quantizer_list(mean, width, frac_width)

    # Generate fixed point representation
    scale = integer_quantizer_list_hw(scale, width, frac_width)
    shift = integer_quantizer_list_hw(shift, width, frac_width)

    arr_mem = {"mean": mean, "var": var}
    if affine:
        gamma = integer_quantizer_list_hw(gamma, width, frac_width)
        beta = integer_quantizer_list_hw(beta, width, frac_width)
        arr_mem.update({"gamma": gamma, "beta": beta})

    # File names with id
    scale_mem_path = mem_dir / f"scale_lutmem-{mem_id}.mem"
    shift_mem_path = mem_dir / f"shift_lutmem-{mem_id}.mem"
    arr_mem_path = mem_dir_arr / f"arr_mem-{mem_id}.mem"

    # Write LUTs to mem files.
    write_memb(scale_mem_path, scale, width)
    write_memb(shift_mem_path, shift, width)
    write_float_mem(arr_mem, arr_mem_path)

    return scale_mem_path, shift_mem_path


@cocotb.test()
async def basic(dut):
    tb = BatchNorm2dTB(dut)
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=2)
    await Timer(100, "us")
    tb.assert_all_monitors_empty()


@cocotb.test()
async def stream(dut):
    tb = BatchNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.setup_test(batches=100)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()

    # Error analysis
    # import json
    # errs = np.stack(tb.output_monitor.error_log).flatten()
    # logger.info("Mean bit-error: %s" % errs.mean())
    # jsonfile = Path(__file__).parent / "data" / f"batch-{tb.IN_WIDTH}.json"
    # with open(jsonfile, 'w') as f:
    #     json.dump({
    #         "mean": errs.mean().item(),
    #         "error": errs.tolist(),
    #     }, f, indent=4)


@cocotb.test()
async def backpressure(dut):
    tb = BatchNorm2dTB(dut)
    await tb.reset()
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    tb.setup_test(batches=100)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_toggle(dut):
    tb = BatchNorm2dTB(dut)
    await tb.reset()
    tb.output_monitor.ready.value = 1
    tb.in_driver.set_valid_prob(0.5)
    tb.setup_test(batches=100)
    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


@cocotb.test()
async def valid_backpressure(dut):
    tb = BatchNorm2dTB(dut)
    await tb.reset()
    tb.in_driver.set_valid_prob(0.5)
    cocotb.start_soon(bit_driver(dut.out_ready, dut.clk, 0.5))
    tb.setup_test(batches=100)

    await Timer(1000, "us")
    assert tb.output_monitor.exp_queue.empty()


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_batch_norm_2d():

    def gen_cfg(
        total_dim0: int = 4,
        total_dim1: int = 4,
        compute_dim0: int = 2,
        compute_dim1: int = 2,
        channels: int = 2,
        in_width: int = 16,
        in_frac_width: int = 8,
        out_width: int = 16,
        out_frac_width: int = 8,
        mem_id: int = 0,
        affine: bool = False,
    ):
        scale_mem_path, shift_mem_path = gen_mem_files(
            mem_id, in_width, in_frac_width, channels, affine
        )

        params = {
            "TOTAL_DIM0": total_dim0,
            "TOTAL_DIM1": total_dim1,
            "COMPUTE_DIM0": compute_dim0,
            "COMPUTE_DIM1": compute_dim1,
            "NUM_CHANNELS": channels,
            "IN_WIDTH": in_width,
            "IN_FRAC_WIDTH": in_frac_width,
            "OUT_WIDTH": out_width,
            "OUT_FRAC_WIDTH": out_frac_width,
            "SCALE_LUT_MEMFILE": verilator_str_param(str(scale_mem_path)),
            "SHIFT_LUT_MEMFILE": verilator_str_param(str(shift_mem_path)),
            "MEM_ID": mem_id,
            "AFFINE": 1 if affine else 0,
        }
        return params

    test_cfgs = [
        # Default
        gen_cfg(),
        # Precision
        gen_cfg(4, 4, 2, 2, 4, 8, 4, 8, 4, 1),
        gen_cfg(4, 4, 2, 2, 4, 4, 2, 4, 2, 2),
        gen_cfg(4, 4, 2, 2, 4, 2, 1, 2, 1, 3),
        # Rectangle
        gen_cfg(4, 6, 2, 2, 4, 16, 8, 16, 8, 4),
        gen_cfg(6, 2, 2, 2, 4, 16, 8, 16, 8, 5),
        gen_cfg(6, 2, 3, 2, 4, 16, 8, 16, 8, 6),
        gen_cfg(4, 6, 2, 3, 4, 16, 8, 16, 8, 7),
        ## Channels
        gen_cfg(4, 4, 2, 2, 5, 16, 8, 16, 8, 8),
        gen_cfg(4, 4, 2, 2, 6, 16, 8, 16, 8, 9),
        gen_cfg(4, 4, 2, 2, 7, 16, 8, 16, 8, 10),
    ]

    mase_runner(
        module_param_list=test_cfgs,
        trace=True,
    )


if __name__ == "__main__":
    test_batch_norm_2d()
