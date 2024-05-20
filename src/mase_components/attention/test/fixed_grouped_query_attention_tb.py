#!/usr/bin/env python3

import os
import math

import torch
from torch import Tensor
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from chop.nn.modules.gqa import repeat_kv
from chop.nn.quantized.modules import GroupedQueryAttentionInteger
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor, ErrorThresholdStreamMonitor
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
        floor=False
    ) -> None:
        super().__init__(embed_dim, num_heads, num_kv_heads, bias, device, dtype,
                         linear_q_config, linear_out_q_config, softermax_out_q_config,
                         qk_matmul_out_q_config, v_matmul_out_q_config, floor)


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

        qk_result = self.qk_matmul_func(
            query_heads,
            key_rep.transpose(2, 3)
        )
        attn_weights = self.softmax_func(qk_result)
        heads_out = self.v_matmul_func(attn_weights, value_rep)

        attn_output = heads_out.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return attn_output, {
            "query": query,
            "key": key.transpose(1, 2),  # Key is transposed in hardware
            "value": value,
            "heads_out": heads_out,
        }


class FixedGroupedQueryAttentionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.assign_self_params([
            "NUM_HEADS",
            "NUM_GROUPS",
            "GROUP_SIZE",
            "ACTIVATION",
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
        ])

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = StreamDriver(
            dut.clk,
            dut.data_in_0,
            dut.data_in_0_valid,
            dut.data_in_0_ready
        )

        # * Weight drivers
        self.weight_q_driver = StreamDriver(
            dut.clk,
            dut.weight_query,
            dut.weight_query_valid,
            dut.weight_query_ready,
        )
        self.weight_k_driver = StreamDriver(
            dut.clk,
            dut.weight_key,
            dut.weight_key_valid,
            dut.weight_key_ready,
        )
        self.weight_v_driver = StreamDriver(
            dut.clk,
            dut.weight_value,
            dut.weight_value_valid,
            dut.weight_value_ready,
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

        self.data_out_0_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            width=self.DATA_OUT_0_PRECISION_0,
            error_bits=1,
            signed=True,
            log_error=True,
            check=False,
            name="Output"
        )

        # Intermediate monitors

        # Q Linear Monitor
        self.query_linear_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.query,
            dut.joint_query_valid,
            dut.joint_query_ready,
            width=self.DATA_OUT_0_PRECISION_0,
            error_bits=0,
            signed=True,
            log_error=True,
            check=False,
            name="Q Linear"
        )

        # K Linear -> Transpose Monitor
        self.key_linear_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.key,
            dut.joint_key_valid,
            dut.joint_key_ready,
            width=self.DATA_OUT_0_PRECISION_0,
            error_bits=0,
            signed=True,
            log_error=True,
            check=False,
            name="K Transpose Linear"
        )

        # V Linear Monitor
        self.value_linear_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.value,
            dut.joint_value_valid,
            dut.joint_value_ready,
            width=self.DATA_OUT_0_PRECISION_0,
            error_bits=0,
            signed=True,
            log_error=True,
            check=False,
            name="V Linear"
        )

        # Head out Monitor
        # TODO: Verilator doesn't support accessing multi-dim arrays
        # self.heads_out_monitor = ErrorThresholdStreamMonitor(
        #     dut.clk,
        #     dut.head_out,
        #     dut.head_out_valid,
        #     dut.head_out_ready,
        #     width=self.DATA_OUT_0_PRECISION_0,
        #     error_bits=0,
        #     signed=True,
        #     log_error=True,
        #     check=False,
        #     name="Heads Out"
        # )


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

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_q_driver.log.setLevel(logging.DEBUG)
        self.weight_k_driver.log.setLevel(logging.DEBUG)
        self.weight_v_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)
        if self.HAS_BIAS:
            self.bias_q_driver.log.setLevel(logging.DEBUG)
            self.bias_k_driver.log.setLevel(logging.DEBUG)
            self.bias_v_driver.log.setLevel(logging.DEBUG)


    def generate_inputs(self, batches=1):
        return torch.randn(
            (
                batches,
                self.DATA_IN_0_TENSOR_SIZE_DIM_1,
                self.DATA_IN_0_TENSOR_SIZE_DIM_0,
            )
        )


    async def run_test(self, batches, us):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out, int_out = self.model(inputs)

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
        for projection in ["q", "k", "v"]:

            projection_name = f"{projection}_projection"

            if self.WEIGHTS_PRE_TRANSPOSED == 1:
                weights = getattr(self.model, projection_name).weight.transpose(0, 1)
            else:
                weights = getattr(self.model, projection_name).weight

            # Normalize K Matrix
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
            self.log.info(f"Loading {len(weights)} beats into weight_{projection}_driver.")
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

        # * Load intermediate monitors
        for projection in ["query", "key", "value"]:
            intermediate_data = int_out[projection]
            monitor = getattr(self, f"{projection}_linear_monitor")
            print(projection, "intermediate_data", intermediate_data.shape)
            mon_data = fixed_preprocess_tensor(
                tensor=intermediate_data,
                q_config={
                    "width": self.DATA_OUT_0_PRECISION_0,
                    "frac_width": self.DATA_OUT_0_PRECISION_1,
                },
                parallelism=[
                    self.DATA_OUT_0_PARALLELISM_DIM_1,
                    self.DATA_OUT_0_PARALLELISM_DIM_0,
                ],
            )
            monitor.load_monitor(mon_data)

        # TODO: Verilator doesn't support accessing multi-dim arrays
        # heads_out = int_out["heads_out"]
        # mon_heads_out = fixed_preprocess_tensor(
        #     tensor=heads_out,
        #     q_config={
        #         "width": self.DATA_OUT_0_PRECISION_0,
        #         "frac_width": self.DATA_OUT_0_PRECISION_1,
        #     },
        #     parallelism=[
        #         self.DATA_OUT_0_PARALLELISM_DIM_1,
        #         self.DATA_OUT_0_PARALLELISM_DIM_0,
        #     ],
        # )
        # self.heads_out_monitor.load_monitor(mon_heads_out)

        # * Load the output monitor
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

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def basic(dut):
    tb = FixedGroupedQueryAttentionTB(dut)
    await tb.run_test(batches=1, us=30)


def get_config(kwargs={}):

    embedding_len = 16
    seq_len = 4
    num_heads = 4
    num_kv_heads = 2

    parallelism = 2
    width = 8
    frac_width = width // 2

    config = {
        "NUM_HEADS": num_heads,
        "NUM_GROUPS": num_kv_heads,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": embedding_len,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": seq_len,
        "DATA_IN_0_PARALLELISM_DIM_0": parallelism,
        "DATA_IN_0_PARALLELISM_DIM_1": parallelism,
        "DATA_IN_0_PRECISION_0": width,
        "DATA_IN_0_PRECISION_1": frac_width,
        "WEIGHTS_PRE_TRANSPOSED": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": embedding_len,
        "WEIGHT_TENSOR_SIZE_DIM_1": embedding_len,
        "WEIGHT_PARALLELISM_DIM_0": parallelism,
        "WEIGHT_PARALLELISM_DIM_1": parallelism,
        "WEIGHT_PRECISION_0": width,
        "WEIGHT_PRECISION_1": frac_width,
        "HAS_BIAS": 0,
        # "BIAS_TENSOR_SIZE_DIM_0": 4,
        # "BIAS_TENSOR_SIZE_DIM_1": 1,
        # "BIAS_PARALLELISM_DIM_0": 2,
        # "BIAS_PARALLELISM_DIM_1": 1,
        # "BIAS_PRECISION_0": 8,
        # "BIAS_PRECISION_1": 4,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": embedding_len,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": seq_len,
        "DATA_OUT_0_PARALLELISM_DIM_0": parallelism,
        "DATA_OUT_0_PARALLELISM_DIM_1": parallelism,
        "DATA_OUT_0_PRECISION_0": width,
        "DATA_OUT_0_PRECISION_1": frac_width,
    }
    config.update(kwargs)
    return config


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    cfgs = [get_config()]
    mase_runner(
        module_param_list=cfgs,
        trace=True,
    )


if __name__ == "__main__":
    test_fixed_linear_smoke()
