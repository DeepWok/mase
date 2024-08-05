#!/usr/bin/env python3

import os
import math
from datetime import datetime
from pathlib import Path
import json
import logging
from functools import partial
from math import ceil

import torch
from torch import Tensor, nn
import numpy as np

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from chop.nn.modules.gqa import repeat_kv
from chop.nn.quantized.modules import GroupedQueryAttentionInteger
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    ErrorThresholdStreamMonitor,
)
from mase_cocotb.runner import mase_runner

from mase_cocotb.utils import fixed_preprocess_tensor


class HardwareGQA(GroupedQueryAttentionInteger):
    """Same as GroupedQueryAttentionInteger but exposes intermediate
    signals on the forward pass so we can compare hardware signals against it."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
        linear_q_config: dict = None,
        linear_out_q_config: dict = None,
        softermax_out_q_config: dict = None,
        qk_matmul_out_q_config: dict = None,
        v_matmul_out_q_config: dict = None,
        floor=False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            device=device,
            dtype=dtype,
            config=linear_q_config,
            out_config=linear_out_q_config,
            floor=floor,
        )

    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape

        query = self.q_projection(x)
        query_heads = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_projection(x)
        key_heads = key.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_projection(x)
        value_heads = value.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        key_rep = repeat_kv(key_heads, n_rep=self.group_size)
        value_rep = repeat_kv(value_heads, n_rep=self.group_size)

        qk_result = self.qk_matmul_func(query_heads, key_rep.transpose(2, 3))
        attn_weights = self.softmax_func(qk_result)
        heads_out = self.v_matmul_func(attn_weights, value_rep)

        attn_output = heads_out.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        out = self.o_projection(attn_output)

        return out, {
            "query": query,
            "key": key.transpose(1, 2),  # Key is transposed in hardware
            "value": value,
            "heads_out": heads_out,
            "attn_output": attn_output,
        }


class FixedGroupedQueryAttentionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst, clk_period_ns=5)

        # global debug switch
        self.log_level = logging.INFO

        # global monitor check switch
        self.check = False

        self.assign_self_params(
            [
                "NUM_HEADS",
                "NUM_GROUPS",
                "GROUP_SIZE",
                "DATA_IN_0_TENSOR_SIZE_DIM_0",
                "DATA_IN_0_TENSOR_SIZE_DIM_1",
                "DATA_IN_0_PARALLELISM_DIM_0",
                "DATA_IN_0_PARALLELISM_DIM_1",
                "DATA_IN_0_PRECISION_0",
                "DATA_IN_0_PRECISION_1",
                "WEIGHTS_PRE_TRANSPOSED",
                "WEIGHT_TENSOR_SIZE_DIM_0",
                "WEIGHT_TENSOR_SIZE_DIM_1",
                "WEIGHT_PARALLELISM_DIM_0",
                "WEIGHT_PARALLELISM_DIM_1",
                "WEIGHT_PRECISION_0",
                "WEIGHT_PRECISION_1",
                "GROUPED_WEIGHT_TENSOR_SIZE_DIM_0",
                "GROUPED_WEIGHT_TENSOR_SIZE_DIM_1",
                "GROUPED_WEIGHT_PARALLELISM_DIM_0",
                "GROUPED_WEIGHT_PARALLELISM_DIM_1",
                "GROUPED_WEIGHT_PRECISION_0",
                "GROUPED_WEIGHT_PRECISION_1",
                "HAS_BIAS",
                "BIAS_TENSOR_SIZE_DIM_0",
                "BIAS_TENSOR_SIZE_DIM_1",
                "BIAS_PARALLELISM_DIM_0",
                "BIAS_PARALLELISM_DIM_1",
                "BIAS_PRECISION_0",
                "BIAS_PRECISION_1",
                "DATA_OUT_0_TENSOR_SIZE_DIM_0",
                "DATA_OUT_0_TENSOR_SIZE_DIM_1",
                "DATA_OUT_0_PARALLELISM_DIM_0",
                "DATA_OUT_0_PARALLELISM_DIM_1",
                "DATA_OUT_0_PRECISION_0",
                "DATA_OUT_0_PRECISION_1",
            ]
        )

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(self.log_level)

        self.data_in_0_driver = StreamDriver(
            dut.clk,
            dut.data_in_0,
            dut.data_in_0_valid,
            dut.data_in_0_ready,
            record_num_beats=True,
        )

        # * Weight drivers
        self.weight_q_driver = StreamDriver(
            dut.clk,
            dut.weight_query,
            dut.weight_query_valid,
            dut.weight_query_ready,
            record_num_beats=True,
        )
        self.weight_k_driver = StreamDriver(
            dut.clk,
            dut.weight_key,
            dut.weight_key_valid,
            dut.weight_key_ready,
            record_num_beats=True,
        )
        self.weight_v_driver = StreamDriver(
            dut.clk,
            dut.weight_value,
            dut.weight_value_valid,
            dut.weight_value_ready,
            record_num_beats=True,
        )
        self.weight_o_driver = StreamDriver(
            dut.clk,
            dut.weight_output,
            dut.weight_output_valid,
            dut.weight_output_ready,
            record_num_beats=True,
        )

        if self.HAS_BIAS == 1:
            self.bias_q_driver = StreamDriver(
                dut.clk,
                dut.bias_query,
                dut.bias_query_valid,
                dut.bias_query_ready,
            )
            self.bias_k_driver = StreamDriver(
                dut.clk,
                dut.bias_key,
                dut.bias_key_valid,
                dut.bias_key_ready,
            )
            self.bias_v_driver = StreamDriver(
                dut.clk,
                dut.bias_value,
                dut.bias_value_valid,
                dut.bias_value_ready,
            )
            self.bias_o_driver = StreamDriver(
                dut.clk,
                dut.bias_output,
                dut.bias_output_valid,
                dut.bias_output_ready,
            )

        self.error_threshold = 2

        self.data_out_0_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            width=self.DATA_OUT_0_PRECISION_0,
            error_bits=self.error_threshold,
            signed=True,
            log_error=True,
            check=self.check,
            name="Output",
        )

        # Model
        linear_q_config = {
            "data_in_width": self.DATA_IN_0_PRECISION_0,
            "data_in_frac_width": self.DATA_IN_0_PRECISION_1,
            "weight_width": self.WEIGHT_PRECISION_0,
            "weight_frac_width": self.WEIGHT_PRECISION_1,
            "bias_width": self.BIAS_PRECISION_0,
            "bias_frac_width": self.BIAS_PRECISION_1,
        }

        # All have same output width configuration
        linear_out_q_config = {
            "data_out_width": self.DATA_OUT_0_PRECISION_0,
            "data_out_frac_width": self.DATA_OUT_0_PRECISION_1,
        }
        qk_matmul_out_q_config = linear_out_q_config
        v_matmul_out_q_config = linear_out_q_config
        softermax_out_q_config = {
            "width": linear_out_q_config["data_out_width"],
            "frac_width": linear_out_q_config["data_out_frac_width"],
        }

        self.model = HardwareGQA(
            embed_dim=self.DATA_IN_0_TENSOR_SIZE_DIM_0,
            num_heads=self.NUM_HEADS,
            num_kv_heads=self.NUM_GROUPS,
            bias=self.HAS_BIAS,
            floor=True,
            linear_q_config=linear_q_config,
            linear_out_q_config=linear_out_q_config,
            qk_matmul_out_q_config=qk_matmul_out_q_config,
            softermax_out_q_config=softermax_out_q_config,
            v_matmul_out_q_config=v_matmul_out_q_config,
        )

        # Init Weights
        # low = -(2**(self.WEIGHT_PRECISION_0-1)) / (2**self.WEIGHT_PRECISION_1)
        # high = ((2**(self.WEIGHT_PRECISION_0-1))-1) / (2**self.WEIGHT_PRECISION_1)
        # nn.init.uniform_(self.model.q_projection.weight, low, high)
        # nn.init.uniform_(self.model.k_projection.weight, low, high)
        # nn.init.uniform_(self.model.v_projection.weight, low, high)
        # nn.init.uniform_(self.model.o_projection.weight, low, high)

        # Set verbosity of driver and monitor loggers to debug
        # self.data_in_0_driver.log.setLevel(logging.DEBUG)
        # self.weight_q_driver.log.setLevel(logging.DEBUG)
        # self.weight_k_driver.log.setLevel(logging.DEBUG)
        # self.weight_v_driver.log.setLevel(logging.DEBUG)
        # self.weight_o_driver.log.setLevel(logging.DEBUG)
        # self.data_out_0_monitor.log.setLevel(logging.DEBUG)
        # if self.HAS_BIAS:
        #     self.bias_q_driver.log.setLevel(logging.DEBUG)
        #     self.bias_k_driver.log.setLevel(logging.DEBUG)
        #     self.bias_v_driver.log.setLevel(logging.DEBUG)
        #     self.bias_o_driver.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batches=1):
        def _shift_dist(x):
            """
            Shifts distribution standard normal input range so that 2 sigma is
            where the MAX_INT & MIN_INT is.
            """
            single_tail_range = 2 ** (
                self.DATA_IN_0_PRECISION_0 - self.DATA_IN_0_PRECISION_1 - 1
            )
            half = single_tail_range / 2
            return x * half

        rand_x = torch.randn(
            (
                batches,
                self.DATA_IN_0_TENSOR_SIZE_DIM_1,
                self.DATA_IN_0_TENSOR_SIZE_DIM_0,
            )
        )

        return _shift_dist(rand_x)

    def _final_check(self):
        max_bit_err = np.max(np.concatenate(self.data_out_0_monitor.error_log))
        self.log.info("Maximum bit-error: %d", max_bit_err)
        if max_bit_err > self.error_threshold:
            assert False, (
                "Test failed due to high approximation error. Got %d bits of error!"
                % max_bit_err
            )

    def _load_inputs_and_weights(self, inputs):
        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs.shape}")
        inputs = fixed_preprocess_tensor(
            tensor=inputs,
            q_config={
                "width": self.DATA_IN_0_PRECISION_0,
                "frac_width": self.DATA_IN_0_PRECISION_1,
            },
            parallelism=[
                self.DATA_IN_0_PARALLELISM_DIM_1,
                self.DATA_IN_0_PARALLELISM_DIM_0,
            ],
        )
        self.log.info(f"Loading {len(inputs)} beats into data_in_0_driver.")
        self.data_in_0_driver.load_driver(inputs)

        # * Load the weights driver
        for projection in ["q", "k", "v", "o"]:
            projection_name = f"{projection}_projection"

            if self.WEIGHTS_PRE_TRANSPOSED == 1:
                weights = getattr(self.model, projection_name).weight.transpose(0, 1)
            else:
                weights = getattr(self.model, projection_name).weight

            # if projection == "k":
            #     weights = weights / math.sqrt(self.model.head_dim)

            self.log.info(f"Processing {projection_name} weights: {weights.shape}")
            weights = fixed_preprocess_tensor(
                tensor=weights,
                q_config={
                    "width": self.WEIGHT_PRECISION_0,
                    "frac_width": self.WEIGHT_PRECISION_1,
                },
                parallelism=[
                    self.WEIGHT_PARALLELISM_DIM_1,
                    self.WEIGHT_PARALLELISM_DIM_0,
                ],
            )
            self.log.info(
                f"Loading {len(weights)} beats into weight_{projection}_driver."
            )
            getattr(self, f"weight_{projection}_driver").load_driver(weights)

            # * Load the bias driver
            if self.HAS_BIAS == 1:
                bias = getattr(self.model, projection_name).bias
                self.log.info(f"Processing {projection_name} bias: {bias}")
                bias = fixed_preprocess_tensor(
                    tensor=bias,
                    q_config={
                        "width": self.BIAS_PRECISION_0,
                        "frac_width": self.BIAS_PRECISION_1,
                    },
                    parallelism=[
                        self.BIAS_PARALLELISM_DIM_1,
                        self.BIAS_PARALLELISM_DIM_0,
                    ],
                )
                getattr(self, f"bias_{projection}_driver").load_driver(bias)

    def _load_outputs(self, exp_out):
        self.log.info(f"Processing outputs: {exp_out.shape}")
        outs = fixed_preprocess_tensor(
            tensor=exp_out,
            q_config={
                "width": self.DATA_OUT_0_PRECISION_0,
                "frac_width": self.DATA_OUT_0_PRECISION_1,
            },
            parallelism=[
                self.DATA_OUT_0_PARALLELISM_DIM_1,
                self.DATA_OUT_0_PARALLELISM_DIM_0,
            ],
        )
        self.log.info(f"Loading {len(outs)} beats into data_out_0_monitor.")
        self.data_out_0_monitor.load_monitor(outs)

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out, int_out = self.model(inputs)

        self._load_inputs_and_weights(inputs)
        self._load_outputs(exp_out)

        if self.DATA_IN_0_PARALLELISM_DIM_1 * self.DATA_IN_0_PARALLELISM_DIM_0 < 4:
            us = 4000
        else:
            us = 1000
        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()

        _, picosec = self.data_out_0_monitor.last_timestamp
        nanosec = picosec / 1000
        clock_period_picosec = self.clock.period

        self.log.info("Entire batch took %f us." % (nanosec / 1000))
        self.log.info("Clock Cycles: %d" % (picosec / clock_period_picosec))
        self.log.info("Clock period: %f ns" % (clock_period_picosec / 1000))

        all_errors = np.concatenate(self.data_out_0_monitor.error_log)
        max_bit_err = np.max(all_errors)
        total_out_size = (
            self.DATA_OUT_0_TENSOR_SIZE_DIM_0 * self.DATA_OUT_0_TENSOR_SIZE_DIM_1
        )
        average_err = np.sum(all_errors) / total_out_size

        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        filename = Path(__file__).parent / f"results/simulation/{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(
                {
                    "seq_len": self.DATA_IN_0_TENSOR_SIZE_DIM_1,
                    "embedding_len": self.DATA_IN_0_TENSOR_SIZE_DIM_0,
                    "seq_paralellism": self.DATA_IN_0_PARALLELISM_DIM_1,
                    "embedding_paralellism": self.DATA_IN_0_PARALLELISM_DIM_0,
                    "num_heads": self.NUM_HEADS,
                    "num_kv_heads": self.NUM_GROUPS,
                    "width": self.DATA_IN_0_PRECISION_0,
                    "frac_width": self.DATA_IN_0_PRECISION_1,
                    "max_err": max_bit_err.item(),
                    "avg_err": average_err.item(),
                    "latency_us": (nanosec / 1000),
                    "clock_cycles": (picosec / clock_period_picosec),
                    "clock_period_ns": (clock_period_picosec / 1000),
                },
                f,
                indent=4,
            )

        self._final_check()

    async def run_memory_bandwidth_test(self, us: int = 500):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        iters = ceil(us * 0.2)
        for _ in range(iters):
            inputs = self.generate_inputs()
            exp_out, int_out = self.model(inputs)
            self._load_inputs_and_weights(inputs)
            self._load_outputs(exp_out)

        await Timer(us, units="us")

        _, picosec = self.data_out_0_monitor.last_timestamp
        nanosec = picosec / 1000
        clock_period_picosec = self.clock.period

        num_input_beats_sent = self.data_in_0_driver.num_beats
        num_q_weight_beats_sent = self.weight_q_driver.num_beats
        num_k_weight_beats_sent = self.weight_k_driver.num_beats
        num_v_weight_beats_sent = self.weight_v_driver.num_beats
        num_o_weight_beats_sent = self.weight_o_driver.num_beats

        input_beats_per_sec = num_input_beats_sent / (nanosec * (10**-9))
        num_q_beats_per_sec = num_q_weight_beats_sent / (nanosec * (10**-9))
        num_k_beats_per_sec = num_k_weight_beats_sent / (nanosec * (10**-9))
        num_v_beats_per_sec = num_v_weight_beats_sent / (nanosec * (10**-9))
        num_o_beats_per_sec = num_o_weight_beats_sent / (nanosec * (10**-9))

        self.log.info("Test length (ns): %.4f" % nanosec)

        self.log.info("Num Input Beats Sent: %d" % num_input_beats_sent)
        self.log.info("Num Q Beats Sent: %d" % num_q_weight_beats_sent)
        self.log.info("Num K Beats Sent: %d" % num_k_weight_beats_sent)
        self.log.info("Num V Beats Sent: %d" % num_v_weight_beats_sent)
        self.log.info("Num Output Beats Sent: %d" % num_o_weight_beats_sent)

        self.log.info("Input Beats per second: %.4f" % input_beats_per_sec)
        self.log.info("Input Q per second: %.4f" % num_q_beats_per_sec)
        self.log.info("Input K per second: %.4f" % num_k_beats_per_sec)
        self.log.info("Input V per second: %.4f" % num_v_beats_per_sec)
        self.log.info("Input O per second: %.4f" % num_o_beats_per_sec)

        all_errors = np.concatenate(self.data_out_0_monitor.error_log)
        max_bit_err = np.max(all_errors)
        total_out_size = (
            iters
            * self.DATA_OUT_0_TENSOR_SIZE_DIM_0
            * self.DATA_OUT_0_TENSOR_SIZE_DIM_1
        )
        average_err = np.sum(all_errors) / total_out_size

        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        filename = Path(__file__).parent / f"results/simulation/{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(
                {
                    "seq_len": self.DATA_IN_0_TENSOR_SIZE_DIM_1,
                    "embedding_len": self.DATA_IN_0_TENSOR_SIZE_DIM_0,
                    "seq_paralellism": self.DATA_IN_0_PARALLELISM_DIM_1,
                    "embedding_paralellism": self.DATA_IN_0_PARALLELISM_DIM_0,
                    "num_heads": self.NUM_HEADS,
                    "num_kv_heads": self.NUM_GROUPS,
                    "width": self.DATA_IN_0_PRECISION_0,
                    "frac_width": self.DATA_IN_0_PRECISION_1,
                    "max_err": max_bit_err.item(),
                    "avg_err": average_err.item(),
                    "test_length_us": (nanosec / 1000),
                    "clock_cycles": (picosec / clock_period_picosec),
                    "clock_period_ns": (clock_period_picosec / 1000),
                    "num_input_beats_sent": num_input_beats_sent,
                    "num_q_weight_beats_sent": num_q_weight_beats_sent,
                    "num_k_weight_beats_sent": num_k_weight_beats_sent,
                    "num_v_weight_beats_sent": num_v_weight_beats_sent,
                    "num_o_weight_beats_sent": num_o_weight_beats_sent,
                },
                f,
                indent=4,
            )


@cocotb.test(skip=True)
async def basic(dut):
    tb = FixedGroupedQueryAttentionTB(dut)
    await tb.run_test()


@cocotb.test()
async def memory_bandwidth(dut):
    tb = FixedGroupedQueryAttentionTB(dut)
    await tb.run_memory_bandwidth_test(us=2000)


def get_config(
    seq_len: int,
    embedding_len: int,
    num_heads: int,
    num_kv_heads: int,
    embedding_parallelism: int,
    sequence_parallelism: int,
    width: int = 8,
    frac_width: int = 4,
):
    return {
        "NUM_HEADS": num_heads,
        "NUM_GROUPS": num_kv_heads,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": embedding_len,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": seq_len,
        "DATA_IN_0_PARALLELISM_DIM_0": embedding_parallelism,
        "DATA_IN_0_PARALLELISM_DIM_1": sequence_parallelism,
        "DATA_IN_0_PRECISION_0": width,
        "DATA_IN_0_PRECISION_1": frac_width,
        "WEIGHTS_PRE_TRANSPOSED": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": embedding_len,
        "WEIGHT_TENSOR_SIZE_DIM_1": embedding_len,
        "WEIGHT_PARALLELISM_DIM_0": embedding_parallelism,
        "WEIGHT_PARALLELISM_DIM_1": embedding_parallelism,
        "WEIGHT_PRECISION_0": width,
        "WEIGHT_PRECISION_1": frac_width,
        "HAS_BIAS": 0,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": embedding_len,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": seq_len,
        "DATA_OUT_0_PARALLELISM_DIM_0": embedding_parallelism,
        "DATA_OUT_0_PARALLELISM_DIM_1": sequence_parallelism,
        "DATA_OUT_0_PRECISION_0": width,
        "DATA_OUT_0_PRECISION_1": frac_width,
    }


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    cfgs = [
        # 4 Groups of 2 heads
        get_config(16, 128, 8, 4, 16, 1),
        # Normal Multi-head Attention 8 QKV heads
        # get_config(10, 128, 8, 8, 2, 2),
        # Multi-Query Attnetion (single head)
        # get_config(10, 128, 8, 1, 2, 2),
        # Mistral-7B (2-bit)
        # get_config(4096, 4096, 32, 8, 4, 4, width=2, frac_width=1),
    ]

    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
    )


def test_parallelism_sweep():
    # Parallelism Sweep
    cfgs = []
    for embedding_par in [1, 2, 4, 8, 16]:
        for seq_par in [1, 2, 4, 8, 16]:
            cfgs.append(get_config(16, 128, 8, 4, embedding_par, seq_par))

    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
    )


def test_small_parallelism():
    # Parallelism Sweep
    cfgs = []
    for embedding_par in [1, 2, 4, 8, 16]:
        for seq_par in [1, 2]:
            cfgs.append(get_config(16, 128, 8, 4, embedding_par, seq_par))

    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
    )


def test_heads_sweep():
    cfgs = []
    for kv_heads in [1, 2, 4, 8, 16]:
        cfgs.append(get_config(256, 256, 16, kv_heads, 16, 1))

    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
    )


def test_bitwidth_sweep():
    cfgs = []
    for bitwidth in range(2, 16 + 1):
        cfgs.append(
            get_config(16, 128, 8, 4, 16, 1, width=bitwidth, frac_width=bitwidth // 2)
        )

    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
    )


def more_realistic():
    cfgs = [
        # get_config(128, 128, 8, 4, 16, 1),  # Works
        # get_config(256, 256, 8, 4, 16, 1),  # Works
        # get_config(512, 512, 8, 4, 16, 1),  # Works
        # get_config(1024, 1024, 8, 4, 16, 1),  # Works
        # get_config(2048, 2048, 8, 4, 16, 1),  # Works
        get_config(4096, 4096, 32, 8, 4, 1, width=2, frac_width=1),
    ]
    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
        extra_build_args=["--unroll-count", "10000"],
    )


def mistral():
    # not possible, verilator will crash
    cfgs = [get_config(4096, 4096, 32, 8, 32, 1)]
    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
        # Needed for large generate loops
        extra_build_args=["--unroll-count", "10000"],
    )


def llama_160m():
    cfgs = []
    for kv_heads in [1, 2, 3, 4, 6, 12]:
        cfgs.append(get_config(2048, 768, 12, kv_heads, 32, 1))
    mase_runner(
        module_param_list=cfgs,
        hierarchical=True,
        template=True,
        sim="verilator",
        extra_build_args=["--unroll-count", "10000"],
    )


if __name__ == "__main__":
    test_fixed_linear_smoke()
    # test_parallelism_sweep()
    # test_small_parallelism()
    # test_heads_sweep()
    # test_bitwidth_sweep()
    # more_realistic()
    # mistral()
    # mqa()
    # llama_160m()
