#!/usr/bin/env python3

import logging
from math import ceil

import torch
import numpy as np

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)
from mase_cocotb.utils import bit_driver, sign_extend_t, batched, signed_to_unsigned
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor

import cocotb
from cocotb.triggers import *

from chop.nn.quantized.functional import fixed_softermax

from chop.nn.quantizers.integer import (
    integer_floor_quantizer,
)
from chop.nn.quantizers.quantizers_for_hw import (
    unsigned_integer_quantizer_for_hw,
)

logger = logging.getLogger("testbench")
logger.setLevel("INFO")


class FixedGQAHeadTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(
            [
                "TOTAL_EMBEDDING_DIM",
                "TOTAL_HEAD_DIM",
                "TOTAL_SEQUENCE_DIM",
                "COMPUTE_EMBEDDING_DIM",
                "COMPUTE_HEAD_DIM",
                "COMPUTE_SEQUENCE_DIM",
                "Q_ACT_WIDTH",
                "Q_ACT_FRAC_WIDTH",
                "Q_WEIGHT_WIDTH",
                "Q_WEIGHT_FRAC_WIDTH",
                "K_ACT_WIDTH",
                "K_ACT_FRAC_WIDTH",
                "V_ACT_WIDTH",
                "V_ACT_FRAC_WIDTH",
                "OUT_ACT_WIDTH",
                "OUT_ACT_FRAC_WIDTH",
                "Q_OUT_WIDTH",
                "Q_OUT_FRAC_WIDTH",
                "QK_OUT_WIDTH",
                "QK_OUT_FRAC_WIDTH",
                "SOFTERMAX_POW2_WIDTH",
                "SOFTERMAX_OUT_WIDTH",
                "SOFTERMAX_OUT_FRAC_WIDTH",
                "EMBEDDING_DEPTH",
                "HEAD_DEPTH",
                "SEQUENCE_DEPTH",
            ]
        )

        # Additional Params
        self.q_act_num_iters = self.EMBEDDING_DEPTH * self.SEQUENCE_DEPTH
        self.q_weight_num_iters = self.HEAD_DEPTH * self.EMBEDDING_DEPTH
        self.k_transpose_num_iters = self.SEQUENCE_DEPTH * self.HEAD_DEPTH
        self.v_act_num_iters = self.HEAD_DEPTH * self.SEQUENCE_DEPTH

        self.q_act_dims = dict(
            total_dim0=self.TOTAL_EMBEDDING_DIM,
            total_dim1=self.TOTAL_SEQUENCE_DIM,
            compute_dim0=self.COMPUTE_EMBEDDING_DIM,
            compute_dim1=self.COMPUTE_SEQUENCE_DIM,
        )
        self.q_weight_dims = dict(
            total_dim0=self.TOTAL_HEAD_DIM,
            total_dim1=self.TOTAL_EMBEDDING_DIM,
            compute_dim0=self.COMPUTE_HEAD_DIM,
            compute_dim1=self.COMPUTE_EMBEDDING_DIM,
        )
        self.k_transpose_act_dims = dict(
            total_dim0=self.TOTAL_SEQUENCE_DIM,
            total_dim1=self.TOTAL_HEAD_DIM,
            compute_dim0=self.COMPUTE_SEQUENCE_DIM,
            compute_dim1=self.COMPUTE_HEAD_DIM,
        )
        self.v_act_dims = dict(
            total_dim0=self.TOTAL_HEAD_DIM,
            total_dim1=self.TOTAL_SEQUENCE_DIM,
            compute_dim0=self.COMPUTE_HEAD_DIM,
            compute_dim1=self.COMPUTE_SEQUENCE_DIM,
        )
        self.out_act_dims = dict(
            total_dim0=self.TOTAL_HEAD_DIM,
            total_dim1=self.TOTAL_SEQUENCE_DIM,
            compute_dim0=self.COMPUTE_HEAD_DIM,
            compute_dim1=self.COMPUTE_SEQUENCE_DIM,
        )

        self.q_act_widths = dict(
            width=self.Q_ACT_WIDTH,
            frac_width=self.Q_ACT_FRAC_WIDTH,
        )
        self.q_weight_widths = dict(
            width=self.Q_WEIGHT_WIDTH,
            frac_width=self.Q_WEIGHT_FRAC_WIDTH,
        )
        self.k_transpose_act_widths = dict(
            width=self.K_ACT_WIDTH,
            frac_width=self.K_ACT_FRAC_WIDTH,
        )
        self.v_act_widths = dict(
            width=self.V_ACT_WIDTH,
            frac_width=self.V_ACT_FRAC_WIDTH,
        )

        # Driver/Monitors
        self.q_act_driver = StreamDriver(
            dut.clk,
            dut.q_act_data,
            dut.q_act_valid,
            dut.q_act_ready,
        )
        self.q_weight_driver = StreamDriver(
            dut.clk,
            dut.q_weight_data,
            dut.q_weight_valid,
            dut.q_weight_ready,
        )
        self.k_tranposed_act_driver = StreamDriver(
            dut.clk,
            dut.k_transposed_act_data,
            dut.k_transposed_act_valid,
            dut.k_transposed_act_ready,
        )
        self.v_act_driver = StreamDriver(
            dut.clk,
            dut.v_act_data,
            dut.v_act_valid,
            dut.v_act_ready,
        )

        # Specify Error Threshold
        self.percentage_error = 0.05  # 5%
        self.error_threshold_bits = ceil(
            self.percentage_error * (2**self.OUT_ACT_WIDTH)
        )

        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.out_act_data,
            dut.out_act_valid,
            dut.out_act_ready,
            width=self.OUT_ACT_WIDTH,
            signed=True,
            error_bits=1,  # 1 bit rounding error
            log_error=True,
            check=True,
        )

    def generate_inputs(self, batches=10):
        q_act = []
        q_weight = []
        k_transpose_act = []
        v_act = []

        for _ in range(batches):
            q_act.extend(
                gen_random_matrix_input(**self.q_act_dims, **self.q_act_widths)
            )
            q_weight.extend(
                gen_random_matrix_input(**self.q_weight_dims, **self.q_weight_widths)
            )
            k_transpose_act.extend(
                gen_random_matrix_input(
                    **self.k_transpose_act_dims, **self.k_transpose_act_widths
                )
            )
            v_act.extend(
                gen_random_matrix_input(**self.v_act_dims, **self.v_act_widths)
            )

        return {
            "q_act": q_act,
            "q_weight": q_weight,
            "k_transpose_act": k_transpose_act,
            "v_act": v_act,
        }

    def model(self, inputs: dict[str, list]):

        def _reconstruct(
            input_list,
            num_iters,
            total_dim0,
            total_dim1,
            compute_dim0,
            compute_dim1,
            width,
            frac_width,
        ):
            matrix_list = []
            for mat in batched(input_list, n=num_iters):
                matrix_list.append(
                    rebuild_matrix(
                        x=mat,
                        total_dim0=total_dim0,
                        total_dim1=total_dim1,
                        compute_dim0=compute_dim0,
                        compute_dim1=compute_dim1,
                    )
                )
            matrix_t = torch.stack(matrix_list)
            signed_matrix = sign_extend_t(matrix_t, bits=width)
            scaled_matrix = signed_matrix.float() / (2**frac_width)
            return scaled_matrix

        q_act = _reconstruct(
            input_list=inputs["q_act"],
            num_iters=self.q_act_num_iters,
            **self.q_act_dims,
            **self.q_act_widths,
        )
        q_weight = _reconstruct(
            input_list=inputs["q_weight"],
            num_iters=self.q_weight_num_iters,
            **self.q_weight_dims,
            **self.q_weight_widths,
        )
        k_transpose_act = _reconstruct(
            input_list=inputs["k_transpose_act"],
            num_iters=self.k_transpose_num_iters,
            **self.k_transpose_act_dims,
            **self.k_transpose_act_widths,
        )
        v_act = _reconstruct(
            input_list=inputs["v_act"],
            num_iters=self.v_act_num_iters,
            **self.v_act_dims,
            **self.v_act_widths,
        )

        logger.debug("q_act: %s" % q_act)
        logger.debug("q_weight: %s" % q_weight)
        logger.debug("k_transpose_act: %s" % k_transpose_act)
        logger.debug("v_act: %s" % v_act)

        q_out = torch.matmul(q_act, q_weight)
        q_out = integer_floor_quantizer(
            x=q_out,
            width=self.Q_OUT_WIDTH,
            frac_width=self.Q_OUT_FRAC_WIDTH,
            is_signed=True,
        )

        qk_out = torch.matmul(q_out, k_transpose_act)
        qk_out = integer_floor_quantizer(
            x=qk_out,
            width=self.QK_OUT_WIDTH,
            frac_width=self.QK_OUT_FRAC_WIDTH,
            is_signed=True,
        )

        softermax_out = fixed_softermax(
            input=qk_out,
            q_config={
                "width": self.QK_OUT_WIDTH,
                "frac_width": self.QK_OUT_FRAC_WIDTH,
            },
            dim=2,
        )
        softermax_out = integer_floor_quantizer(
            x=softermax_out,
            width=self.SOFTERMAX_OUT_WIDTH,
            frac_width=self.SOFTERMAX_OUT_FRAC_WIDTH,
            is_signed=False,
        )

        attention_out = torch.matmul(softermax_out, v_act)
        attention_out = integer_floor_quantizer(
            x=attention_out,
            width=self.OUT_ACT_WIDTH,
            frac_width=self.OUT_ACT_FRAC_WIDTH,
            is_signed=True,
        )

        logger.debug("q_out: %s" % q_out)
        logger.debug("qk_out: %s" % qk_out)
        logger.debug("softermax_out: %s" % softermax_out)
        logger.debug("attention_out: %s" % attention_out)

        # Process output
        rounded_atten = integer_floor_quantizer(
            x=attention_out,
            width=self.OUT_ACT_WIDTH,
            frac_width=self.OUT_ACT_FRAC_WIDTH,
            is_signed=True,
        )
        atten_int = (rounded_atten * (2**self.OUT_ACT_FRAC_WIDTH)).int()
        atten_uint = signed_to_unsigned(atten_int, bits=self.OUT_ACT_WIDTH)
        logger.debug("rounded_atten: %s" % rounded_atten)
        logger.debug("atten_int: %s" % atten_int)
        logger.debug("atten_uint: %s" % atten_uint)

        exp_out = []
        for output_matrix in atten_uint:
            exp_out.extend(split_matrix(output_matrix, **self.out_act_dims))
        return exp_out

    async def run_test(self, batches, us):
        inputs = self.generate_inputs(batches)
        # Load Drivers
        self.q_act_driver.load_driver(inputs["q_act"])
        self.q_weight_driver.load_driver(inputs["q_weight"])
        self.k_tranposed_act_driver.load_driver(inputs["k_transpose_act"])
        self.v_act_driver.load_driver(inputs["v_act"])
        # Get expectation from model
        exp_out = self.model(inputs)
        self.output_monitor.load_monitor(exp_out)
        await Timer(us, "us")
        assert self.output_monitor.recv_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = FixedGQAHeadTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=3, us=10)


@cocotb.test()
async def stream(dut):
    tb = FixedGQAHeadTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.reset()
    await tb.run_test(batches=200, us=2000)


@cocotb.test()
async def backpressure(dut):
    tb = FixedGQAHeadTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    await tb.reset()
    await tb.run_test(batches=200, us=2000)


@cocotb.test()
async def valid(dut):
    tb = FixedGQAHeadTB(dut)
    tb.output_monitor.ready.value = 1
    tb.q_act_driver.set_valid_prob(0.5)
    tb.q_weight_driver.set_valid_prob(0.5)
    tb.k_tranposed_act_driver.set_valid_prob(0.5)
    tb.v_act_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=200, us=2000)


@cocotb.test()
async def valid_backpressure(dut):
    tb = FixedGQAHeadTB(dut)
    cocotb.start_soon(bit_driver(tb.output_monitor.ready, tb.clk, 0.5))
    tb.q_act_driver.set_valid_prob(0.5)
    tb.q_weight_driver.set_valid_prob(0.5)
    tb.k_tranposed_act_driver.set_valid_prob(0.5)
    tb.v_act_driver.set_valid_prob(0.5)
    await tb.reset()
    await tb.run_test(batches=200, us=2000)


if __name__ == "__main__":

    def width_cfgs(prefix: str, cfgs: list[dict]):
        new_cfgs = []
        for cfg in cfgs:
            new_cfgs.append({**cfg, f"{prefix}_WIDTH": 8, f"{prefix}_FRAC_WIDTH": 4})
            new_cfgs.append({**cfg, f"{prefix}_WIDTH": 16, f"{prefix}_FRAC_WIDTH": 8})
        return new_cfgs

    def dimension_cfgs(cfgs: list[dict]):
        new_cfgs = []
        for cfg in cfgs:
            new_cfgs.append(
                {
                    **cfg,
                    "TOTAL_EMBEDDING_DIM": 64,
                    "TOTAL_HEAD_DIM": 16,
                    "TOTAL_SEQUENCE_DIM": 8,
                }
            )
            new_cfgs.append(
                {
                    **cfg,
                    "TOTAL_EMBEDDING_DIM": 16,
                    "TOTAL_HEAD_DIM": 8,
                    "TOTAL_SEQUENCE_DIM": 16,
                }
            )
        return new_cfgs

    def compute_dim_cfgs(cfgs: list[dict]):
        new_cfgs = []
        for cfg in cfgs:
            new_cfgs.append(
                {
                    **cfg,
                    "COMPUTE_EMBEDDING_DIM": 2,
                    "COMPUTE_HEAD_DIM": 2,
                    "COMPUTE_SEQUENCE_DIM": 2,
                }
            )
            new_cfgs.append(
                {
                    **cfg,
                    "COMPUTE_EMBEDDING_DIM": 4,
                    "COMPUTE_HEAD_DIM": 4,
                    "COMPUTE_SEQUENCE_DIM": 4,
                }
            )
        return new_cfgs

    DEFAULT = {
        # Dimensions
        "TOTAL_EMBEDDING_DIM": 16,
        "TOTAL_HEAD_DIM": 4,
        "TOTAL_SEQUENCE_DIM": 4,  # Number of tokens
        "COMPUTE_EMBEDDING_DIM": 2,
        "COMPUTE_HEAD_DIM": 2,
        "COMPUTE_SEQUENCE_DIM": 2,
        # Input Widths
        "Q_ACT_WIDTH": 8,
        "Q_ACT_FRAC_WIDTH": 4,
        "Q_WEIGHT_WIDTH": 8,
        "Q_WEIGHT_FRAC_WIDTH": 4,
        "K_ACT_WIDTH": 8,
        "K_ACT_FRAC_WIDTH": 2,
        "V_ACT_WIDTH": 8,
        "V_ACT_FRAC_WIDTH": 4,
        # Output Widths
        "OUT_ACT_WIDTH": 8,
        "OUT_ACT_FRAC_WIDTH": 2,
        # Intermediate Widths
        "Q_OUT_WIDTH": 16,
        "Q_OUT_FRAC_WIDTH": 4,
        "QK_OUT_WIDTH": 16,
        "QK_OUT_FRAC_WIDTH": 4,
        "SOFTERMAX_POW2_WIDTH": 16,
        "SOFTERMAX_OUT_WIDTH": 16,
        "SOFTERMAX_OUT_FRAC_WIDTH": 15,
    }

    cfgs = [DEFAULT]
    cfgs = dimension_cfgs(cfgs)
    cfgs = compute_dim_cfgs(cfgs)
    for prefix in ["Q_ACT", "Q_WEIGHT", "K_ACT", "V_ACT", "OUT_ACT"]:
        cfgs = width_cfgs(prefix, cfgs)

    cfgs = [
        {
            "TOTAL_EMBEDDING_DIM": 64,
            "TOTAL_HEAD_DIM": 16,
            "TOTAL_SEQUENCE_DIM": 8,
            "COMPUTE_EMBEDDING_DIM": 2,
            "COMPUTE_HEAD_DIM": 2,
            "COMPUTE_SEQUENCE_DIM": 2,
            "Q_ACT_WIDTH": 8,
            "Q_ACT_FRAC_WIDTH": 4,
            "Q_WEIGHT_WIDTH": 8,
            "Q_WEIGHT_FRAC_WIDTH": 4,
            "K_ACT_WIDTH": 8,
            "K_ACT_FRAC_WIDTH": 4,
            "V_ACT_WIDTH": 8,
            "V_ACT_FRAC_WIDTH": 4,
            "OUT_ACT_WIDTH": 8,
            "OUT_ACT_FRAC_WIDTH": 4,
            "Q_OUT_WIDTH": 16,
            "Q_OUT_FRAC_WIDTH": 4,
            "QK_OUT_WIDTH": 16,
            "QK_OUT_FRAC_WIDTH": 4,
            "SOFTERMAX_POW2_WIDTH": 16,
            "SOFTERMAX_OUT_WIDTH": 16,
            "SOFTERMAX_OUT_FRAC_WIDTH": 15,
        }
    ]
    print(f"Running Tests on {len(cfgs)} Configs...")

    mase_runner(
        module_param_list=cfgs,
        # trace=True,
        jobs=12,
    )
