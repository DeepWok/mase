#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
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
from z_qlayers import quantize_to_int as q2i
from chop.models.manual.quant_utils import get_quantized_cls
from chop.passes.transforms.quantize.quantizers.integer import _integer_quantize

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 8
        self.data_in_frac_width = 1
        self.weight_width = 6
        self.weight_frac_width = 1
        self.data_out_width = 8
        self.data_out_frac_width = 1

        self.has_bias = 1
        self.bias_width = 6
        self.bias_frac_width = 1

        self.in_y = 1
        self.unroll_in_y = 1

        self.in_x = 2
        self.unroll_in_x = 2

        self.w_y = 1
        self.unroll_w_y = 1

        self.in_num_parallelism = int(self.in_y / self.unroll_in_y)
        self.w_num_parallelism = int(self.w_y / self.unroll_w_y)
        self.in_depth = int(self.in_x / self.unroll_in_x)
        _, d, w, b = self.data_generate()
        self.bias = RandomSource(
            name="bias",
            samples=samples * self.w_num_parallelism,
            num=self.unroll_w_y,
            max_stalls=2 * samples * self.w_num_parallelism,
            data_specify=b,
            debug=debug,
        )
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.unroll_in_y * self.unroll_in_x,
            max_stalls=2 * samples * self.in_depth * self.in_num_parallelism,
            data_specify=d,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.unroll_w_y * self.unroll_in_x,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            data_specify=w,
            debug=debug,
        )

        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.w_num_parallelism,
            max_stalls=2 * samples * self.in_num_parallelism * self.w_num_parallelism,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "IN_FRAC_WIDTH": self.data_in_frac_width,
            "WEIGHT_WIDTH": self.weight_width,
            "WEIGHT_FRAC_WIDTH": self.weight_frac_width,
            "HAS_BIAS": self.has_bias,
            "BIAS_WIDTH": self.bias_width,
            "BIAS_FRAC_WIDTH": self.bias_frac_width,
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN_Y": self.in_y,
            "UNROLL_IN_Y": self.unroll_in_y,
            "IN_X": self.in_x,
            "UNROLL_IN_X": self.unroll_in_x,
            "W_Y": self.w_y,
            "UNROLL_W_Y": self.unroll_w_y,
        }

    def data_generate(self):
        torch.manual_seed(0)
        in_features = self.in_x
        out_features = self.w_y
        config = {
            "fc1_proj": {
                "name": "integer",
                "weight_width": self.weight_width,
                "weight_frac_width": self.weight_frac_width,
                "data_in_width": self.data_in_width,
                "data_in_frac_width": self.data_in_frac_width,
                "bias_width": self.bias_width,
                "bias_frac_width": self.bias_frac_width,
            }
        }
        torch.manual_seed(0)
        x = 5 * torch.randn(self.samples, self.in_y, in_features)
        bias = True if self.has_bias == 1 else False
        self.linear = get_quantized_cls("linear", config["fc1_proj"])(
            in_features, out_features, bias=bias, config=config["fc1_proj"]
        )
        print(
            "data_in = ",
            _integer_quantize(x, self.data_in_width, self.data_in_frac_width),
        )
        print(
            "weight  = ",
            _integer_quantize(
                self.linear.weight, self.weight_width, self.weight_frac_width
            ),
        )
        if bias:
            print(
                "bias    = ",
                _integer_quantize(
                    self.linear.bias, self.bias_width, self.bias_frac_width
                ),
            )

        w = q2i(self.linear.weight, self.weight_width, self.weight_frac_width)
        bbb = self.linear.bias if bias else torch.randn(self.in_y)
        b = q2i(bbb, self.bias_width, self.bias_frac_width)

        data_in = self.data_pack(
            q2i(x, self.data_in_width, self.data_in_frac_width),
            np=int(self.in_y / self.unroll_in_y),
            d=int(self.in_x / self.unroll_in_x),
            p=self.unroll_in_y,
            s=self.unroll_in_x,
        )

        weight_in = self.data_pack(
            w.repeat(self.samples, 1, 1),
            np=int(self.w_y / self.unroll_w_y),
            d=int(self.in_x / self.unroll_in_x),
            p=self.unroll_w_y,
            s=self.unroll_in_x,
        )

        bias_in = self.data_pack(
            b.repeat(self.samples, 1),
            np=int(self.w_y / self.unroll_w_y),
            d=1,
            p=self.unroll_w_y,
            s=1,
        )
        logger.debug(
            "\n\
        data_tensor = {} \n\
        weight_tensor = {} \n\
        bias_tensor = {} \n\
        data_in = {} \n\
        weight_in = {} \n\
        bias_in = {} ".format(
                x,
                w,
                b,
                data_in,
                weight_in,
                bias_in,
            )
        )
        data_in.reverse()
        weight_in.reverse()
        bias_in.reverse()
        # NOTE: weight in should transpose here
        return (x, data_in, weight_in, bias_in)

    def data_pack(self, in_temp, np, d, p, s):
        ## just what to make a matrix with [np*p][s*d] to tile [np*d][p*s]
        ## assume the in_temp as torch.float
        in_temp = in_temp.to(torch.int).reshape(self.samples, np * p, d * s)
        ref = []
        for i in range(self.samples):
            re_tensor = rearrange(
                in_temp[i], "(np p) (d s) -> np (p d) s", np=np, d=d, p=p, s=s
            )
            ex_tensor = torch.zeros(np, d * p, s, dtype=int)
            for b in range(np):
                for i in range(d):
                    for j in range(p):
                        ex_tensor[b][i * p + j] = re_tensor[b][j * d + i]
            output_tensor = rearrange(
                ex_tensor, "np (d p) s -> (np d) (p s)", np=np, d=d, p=p, s=s
            )
            output = output_tensor.tolist()
            ref = ref + output
        return ref

    def sw_compute(self):
        x, _, _, _ = self.data_generate()
        data_out = self.linear(x)
        print(data_out)
        output = self.data_pack(
            q2i(data_out, self.data_out_width, self.data_out_frac_width),
            np=int(self.in_y / self.unroll_in_y),
            d=int(self.w_y / self.unroll_w_y),
            p=self.unroll_in_y,
            s=self.unroll_w_y,
        )
        return output


# Check if an is_impossible state is reached
def is_impossible_state(
    weight_ready,
    weight_valid,
    data_in_ready,
    data_in_valid,
    data_out_ready,
    data_out_valid,
):
    return False


def debug_state(dut, state):
    logger.debug(
        "{} State: (weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 20
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
    dut.weight_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    logger.debug(
        "data_in = {}\n\
        weight = {}\n\
        ".format(
            [int(i[0]) for i in test_case.data_in.data],
            [int(i[0]) for i in test_case.weight.data],
        )
    )
    for i in range(samples * 200):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        dut.bias_valid.value = test_case.bias.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        # start input data
        #
        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )

        await Timer(1, units="ns")

        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        logger.debug(
            "wave_check:\n\
            {},{} data_in = {}\n\
            {},{} weight = {}\n\
            {},{} weight = {}\n\
            {},{} weight = {}\n\
            {},{} out_not_cast = {}\n\
            ".format(
                dut.inst_fmmc.inst_fmmc.data_in1_valid.value,
                dut.inst_fmmc.inst_fmmc.data_in1_ready.value,
                [i for i in dut.inst_fmmc.inst_fmmc.data_in1.value],
                dut.weight_valid.value,
                dut.weight_ready.value,
                [i for i in dut.weight.value],
                dut.inst_fmmc.data_in2_valid.value,
                dut.inst_fmmc.data_in2_ready.value,
                [i for i in dut.inst_fmmc.data_in2.value],
                dut.inst_fmmc.inst_fmmc.data_in2_valid.value,
                dut.inst_fmmc.inst_fmmc.data_in2_ready.value,
                [i for i in dut.inst_fmmc.inst_fmmc.data_in2.value],
                dut.inst_fmmc.inst_fmmc.data_out_valid.value,
                dut.inst_fmmc.inst_fmmc.data_out_ready.value,
                [i for i in dut.inst_fmmc.inst_fmmc.cast_data.value],
            )
        )
        # breakpoint()
        if (
            test_case.weight.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/cast/fixed_cast.sv",
        "../../../../components/fixed_arith/fixed_matmul_core.sv",
        "../../../../components/fixed_arith/fixed_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/fixed_arith/fixed_vector_mult.sv",
        "../../../../components/fixed_arith/fixed_adder_tree.sv",
        "../../../../components/fixed_arith/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arith/fixed_mult.sv",
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
        hdl_toplevel="fixed_2d_linear",
        build_args=extra_args,
    )
    runner.test(
        hdl_toplevel="fixed_2d_linear",
        test_module="fixed_2d_linear_tb",
    )


if __name__ == "__main__":
    runner()
