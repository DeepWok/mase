#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/workspace/components/testbench/ViT/")
sys.path.append("/workspace/machop/")

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
from pvt_quant import QuantizedMlp
from z_qlayers import quantize_to_int as q2i
from z_qlayers import linear_data_pack

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 8
        self.data_in_frac_width = 3
        self.weight_i2h_width = 6
        self.weight_i2h_frac_width = 3
        self.weight_h2o_width = 6
        self.weight_h2o_frac_width = 3
        self.has_bias = 1
        self.bias_i2h_width = 6
        self.bias_i2h_frac_width = 4
        self.bias_h2o_width = 6
        self.bias_h2o_frac_width = 4
        self.hidden_width = 8
        self.hidden_frac_width = 4
        self.data_out_width = 8
        self.data_out_frac_width = 4
        self.w_config = {
            "mlp": {
                "fc1_proj": {
                    "name": "integer",
                    "weight_width": self.weight_i2h_width,
                    "weight_frac_width": self.weight_i2h_frac_width,
                    "data_in_width": self.data_in_width,
                    "data_in_frac_width": self.data_in_frac_width,
                    "bias_width": self.bias_i2h_width,
                    "bias_frac_width": self.bias_i2h_frac_width,
                },
                "fc2_proj": {
                    "name": "integer",
                    "weight_width": self.weight_h2o_width,
                    "weight_frac_width": self.weight_h2o_frac_width,
                    "data_in_width": self.data_in_width,
                    "data_in_frac_width": self.data_in_frac_width,
                    "bias_width": self.bias_h2o_width,
                    "bias_frac_width": self.bias_h2o_frac_width,
                },
                "mlp_relu": {
                    "name": "integer",
                    "bypass": True,
                    "data_in_width": self.data_in_width,
                    "data_in_frac_width": self.data_in_frac_width,
                },
            },
        }
        self.in_num = 2
        self.in_features = 16
        self.hidden_features = 2 * self.in_features
        self.out_features = self.in_features

        self.tile_in_num = 1
        self.tile_in_features = 2
        self.tile_hidden_features = 1
        self.tile_out_features = self.tile_in_features
        self.d_config = {
            "mlp": {
                "in_num": self.in_num,
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.in_features,
                "unroll_in_num": self.tile_in_num,
                "unroll_in_features": self.tile_in_features,
                "unroll_hidden_features": self.tile_hidden_features,
                "unroll_out_features": self.tile_out_features,
            },
        }

        self.data_generate()
        depth_in_num = int(self.in_num / self.tile_in_num)
        depth_out_features = int(self.out_features / self.tile_out_features)
        self.outputs = RandomSink(
            samples=samples * depth_out_features * depth_in_num,
            debug=debug,
        )
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["mlp"]["fc1_proj"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["mlp"]["fc1_proj"]["data_in_frac_width"],
            "WEIGHT_I2H_WIDTH": self.w_config["mlp"]["fc1_proj"]["weight_width"],
            "WEIGHT_I2H_FRAC_WIDTH": self.w_config["mlp"]["fc1_proj"][
                "weight_frac_width"
            ],
            "BIAS_I2H_WIDTH": self.w_config["mlp"]["fc1_proj"]["bias_width"],
            "BIAS_I2H_FRAC_WIDTH": self.w_config["mlp"]["fc1_proj"]["bias_frac_width"],
            "HAS_BIAS": self.has_bias,
            "HIDDEN_WIDTH": self.w_config["mlp"]["fc2_proj"]["data_in_width"],
            "HIDDEN_FRAC_WIDTH": self.w_config["mlp"]["fc2_proj"]["data_in_frac_width"],
            "WEIGHT_H2O_WIDTH": self.w_config["mlp"]["fc2_proj"]["weight_width"],
            "WEIGHT_H2O_FRAC_WIDTH": self.w_config["mlp"]["fc2_proj"][
                "weight_frac_width"
            ],
            "BIAS_H2O_WIDTH": self.w_config["mlp"]["fc2_proj"]["bias_width"],
            "BIAS_H2O_FRAC_WIDTH": self.w_config["mlp"]["fc2_proj"]["bias_frac_width"],
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN_NUM": self.in_num,
            "IN_FEATURES": self.in_features,
            "HIDDEN_FEATURES": self.hidden_features,
            "UNROLL_IN_NUM": self.tile_in_num,
            "UNROLL_IN_FEATURES": self.tile_in_features,
            "UNROLL_HIDDEN_FEATURES": self.tile_hidden_features,
            "UNROLL_OUT_FEATURES": self.tile_out_features,
        }

    def data_generate(self):
        torch.manual_seed(0)
        w_config = self.w_config["mlp"]
        self.mlp = QuantizedMlp(
            in_features=self.d_config["mlp"]["in_features"],
            hidden_features=self.d_config["mlp"]["hidden_features"],
            drop=0.0,
            config=self.w_config["mlp"],
        )
        self.x = 5 * torch.randn((self.samples, self.in_num, self.in_features))
        weight1_tensor = q2i(
            self.mlp.fc1.weight,
            w_config["fc1_proj"]["weight_width"],
            w_config["fc1_proj"]["weight_frac_width"],
        )

        bias1_tensor = q2i(
            self.mlp.fc1.bias,
            w_config["fc1_proj"]["bias_width"],
            w_config["fc1_proj"]["bias_frac_width"],
        )

        weight2_tensor = q2i(
            self.mlp.fc2.weight,
            w_config["fc2_proj"]["weight_width"],
            w_config["fc2_proj"]["weight_frac_width"],
        )
        bias2_tensor = q2i(
            self.mlp.fc2.bias,
            w_config["fc2_proj"]["bias_width"],
            w_config["fc2_proj"]["bias_frac_width"],
        )
        x_tensor = q2i(
            self.x,
            w_config["fc1_proj"]["data_in_width"],
            w_config["fc1_proj"]["data_in_frac_width"],
        )
        self.inputs = linear_data_pack(
            self.samples,
            x_tensor,
            self.in_num,
            self.in_features,
            self.tile_in_num,
            self.tile_in_features,
        )
        self.weight1_in = linear_data_pack(
            self.samples,
            weight1_tensor.repeat(self.samples, 1, 1),
            self.hidden_features,
            self.in_features,
            self.tile_hidden_features,
            self.tile_in_features,
        )
        self.bias1_in = linear_data_pack(
            self.samples,
            bias1_tensor.repeat(self.samples, 1, 1),
            self.hidden_features,
            1,
            self.tile_hidden_features,
            1,
        )
        self.weight2_in = linear_data_pack(
            self.samples,
            weight2_tensor.repeat(self.samples, 1, 1),
            self.out_features,
            self.hidden_features,
            self.tile_out_features,
            self.tile_hidden_features,
        )
        self.bias2_in = linear_data_pack(
            self.samples,
            bias2_tensor.repeat(self.samples, 1, 1),
            self.out_features,
            1,
            self.tile_out_features,
            1,
        )
        self.inputs.reverse()
        self.weight1_in.reverse()
        self.bias1_in.reverse()
        self.weight2_in.reverse()
        self.bias2_in.reverse()
        samples = self.samples
        depth_in_features = int(self.in_features / self.tile_in_features)
        depth_in_num = int(self.in_num / self.tile_in_num)
        depth_hidden_features = int(self.hidden_features / self.tile_hidden_features)
        depth_out_features = int(self.out_features / self.tile_out_features)
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * depth_in_features * depth_in_num,
            num=self.tile_in_features * self.tile_in_num,
            max_stalls=2 * samples,
            data_specify=self.inputs,
            debug=debug,
        )
        self.bias1 = RandomSource(
            name="bias1",
            samples=samples * depth_hidden_features,
            num=self.tile_hidden_features,
            max_stalls=2 * samples * depth_hidden_features,
            data_specify=self.bias1_in,
            debug=debug,
        )
        self.bias2 = RandomSource(
            name="bias2",
            samples=samples * depth_out_features,
            num=self.tile_out_features,
            max_stalls=2 * samples * depth_out_features,
            data_specify=self.bias2_in,
            debug=debug,
        )
        self.weight1 = RandomSource(
            name="weight1",
            samples=samples * depth_hidden_features * depth_in_features,
            num=self.tile_hidden_features * self.tile_in_features,
            max_stalls=2 * samples * depth_hidden_features * depth_in_features,
            data_specify=self.weight1_in,
            debug=debug,
        )
        self.weight2 = RandomSource(
            name="weight2",
            samples=samples * depth_out_features * depth_hidden_features,
            num=self.tile_out_features * self.tile_hidden_features,
            max_stalls=2 * samples * depth_hidden_features * depth_out_features,
            data_specify=self.weight2_in,
            debug=debug,
        )

    def sw_compute(self):
        data_out = self.mlp(self.x)
        output = linear_data_pack(
            self.samples,
            q2i(data_out, self.data_out_width, self.data_out_frac_width),
            self.in_num,
            self.out_features,
            self.tile_in_num,
            self.tile_out_features,
        )
        return output


def debug_state(dut, state):
    logger.debug(
        "{} State: (data_in2_ready,data_in2_valid,data_in1_ready,data_in1_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 10
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Synchronize with the clock
    dut.data_in_valid.value = 0
    dut.weight_in2hidden_valid.value = 0
    dut.weight_hidden2out_valid.value = 0
    dut.bias_in2hidden_valid.value = 0
    dut.bias_hidden2out_valid.value = 0
    dut.data_out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    # breakpoint()
    cdin = 0
    cdata_out = 0
    chidden_data = 0
    for i in range(samples * 8000):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.weight_in2hidden_valid.value = test_case.weight1.pre_compute()
        dut.bias_in2hidden_valid.value = test_case.bias1.pre_compute()
        dut.weight_hidden2out_valid.value = test_case.weight2.pre_compute()
        dut.bias_hidden2out_valid.value = test_case.bias2.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")

        # start input data
        #
        (
            dut.weight_in2hidden_valid.value,
            dut.weight_in2hidden.value,
        ) = test_case.weight1.compute(dut.weight_in2hidden_ready.value)
        (
            dut.weight_hidden2out_valid.value,
            dut.weight_hidden2out.value,
        ) = test_case.weight2.compute(dut.weight_hidden2out_ready.value)
        (
            dut.bias_in2hidden_valid.value,
            dut.bias_in2hidden.value,
        ) = test_case.bias1.compute(dut.bias_in2hidden_ready.value)
        (
            dut.bias_hidden2out_valid.value,
            dut.bias_hidden2out.value,
        ) = test_case.bias2.compute(dut.bias_hidden2out_ready.value)
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )

        await Timer(1, units="ns")

        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        wave_check(dut)
        if dut.data_in_valid.value == 1 and dut.data_in_ready.value == 1:
            cdin += 1
        if dut.data_out_valid.value == 1 and dut.data_out_ready.value == 1:
            cdata_out += 1
        if dut.hidden_data_valid.value == 1 and dut.hidden_data_ready.value == 1:
            chidden_data += 1
        print("cdin = ", cdin)
        print("cdata_out = ", cdata_out)
        print("chidden_data = ", chidden_data)
        if (
            # test_case.weight1.is_empty()
            # and test_case.bias1.is_empty()
            # and test_case.weight2.is_empty()
            # and test_case.bias2.is_empty()
            test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of in_out:\n\
            {},{},data_in = {} \n\
            {},{},weight_in2hidden = {} \n\
            {},{},hidden_data = {} \n\
            {},{},relu_data = {} \n\
            {},{},data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.weight_in2hidden_valid.value,
            dut.weight_in2hidden_ready.value,
            [int(i) for i in dut.weight_in2hidden.value],
            dut.hidden_data_valid.value,
            dut.hidden_data_ready.value,
            [int(i) for i in dut.hidden_data.value],
            dut.relu_data_valid.value,
            dut.relu_data_ready.value,
            [int(i) for i in dut.relu_data.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/fixed_mlp.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/activations/fixed_relu.sv",
        "../../../../components/fixed_arithmetic/fixed_matmul_core.sv",
        "../../../../components/fixed_arithmetic/fixed_dot_product.sv",
        "../../../../components/fixed_arithmetic/fixed_accumulator.sv",
        "../../../../components/fixed_arithmetic/fixed_vector_mult.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arithmetic/fixed_mult.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_mlp",
        build_args=extra_args,
    )
    for _ in range(1):
        runner.test(
            hdl_toplevel="fixed_mlp",
            test_module="fixed_mlp_tb",
        )


if __name__ == "__main__":
    runner()
