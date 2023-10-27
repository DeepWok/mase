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

sys.path.append("/workspace/components/testbench/ViT/")
from pvt_quant import QuantizedPatchEmbed

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        # width config
        self.w_config = {
            "patch_proj": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 5,
                "weight_width": 8,
                "weight_frac_width": 6,
                "bias_width": 8,
                "bias_frac_width": 5,
            }
        }
        self.data_out_width = 6
        self.data_out_frac_width = 2
        # parameters config
        self.in_c = 3
        self.in_y = 32
        self.in_x = 32
        self.patch_size = 8
        self.embed_dim = 64
        self.pe_unroll_kernel_out = 3
        self.pe_unroll_in_c = 3
        self.pe_unroll_embed_dim = 8
        self.num_patch = int(self.in_y * self.in_x // (self.patch_size**2))

        # self.num_classes = 10
        # self.head_unroll_out_x = 5
        self.samples = samples

        self.pe_iter_weight = int(
            (self.patch_size**2)
            * self.in_c
            * self.embed_dim
            / self.pe_unroll_kernel_out
            / self.pe_unroll_embed_dim
        )
        self.data_generate()
        # TODO: here
        self.data_in = RandomSource(
            name="data_in",
            samples=samples
            * int(self.in_x * self.in_y * self.in_c / self.pe_unroll_in_c),
            num=self.pe_unroll_in_c,
            max_stalls=0,
            # max_stalls=2 * samples * int(self.in_x * self.in_y * self.in_c / self.pe_unroll_in_c),
            data_specify=self.x_in,
            debug=debug,
        )
        self.patch_embed_bias = RandomSource(
            name="patch_embed_bias",
            samples=samples * int(self.embed_dim / self.pe_unroll_embed_dim),
            num=self.pe_unroll_embed_dim,
            # max_stalls=2 * samples * int(self.embed_dim/self.pe_unroll_embed_dim),
            data_specify=self.pe_b_in,
            debug=debug,
        )
        self.patch_embed_weight = RandomSource(
            name="patch_embed_weight",
            samples=samples * self.pe_iter_weight,
            num=self.pe_unroll_kernel_out * self.pe_unroll_embed_dim,
            # max_stalls=2 * samples * self.pe_iter_weight,
            data_specify=self.pe_w_in,
            debug=debug,
        )

        self.outputs = RandomSink(
            samples=samples
            * self.num_patch
            * int(self.embed_dim / self.pe_unroll_embed_dim),
            max_stalls=0,
            # max_stalls=2 * samples * int(self.num_classes/self.head_unroll_out_x),
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["patch_proj"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["patch_proj"]["data_in_frac_width"],
            "W_WIDTH": self.w_config["patch_proj"]["weight_width"],
            "W_FRAC_WIDTH": self.w_config["patch_proj"]["weight_frac_width"],
            "BIAS_WIDTH": self.w_config["patch_proj"]["bias_width"],
            "BIAS_FRAC_WIDTH": self.w_config["patch_proj"]["bias_frac_width"],
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN_C": self.in_c,
            "IN_Y": self.in_y,
            "IN_X": self.in_x,
            "KERNEL_SIZE": self.patch_size,
            "OUT_C": self.embed_dim,
            "SLIDING_NUM": self.num_patch,
            "UNROLL_KERNEL_OUT": self.pe_unroll_kernel_out,
            "UNROLL_IN_C": self.pe_unroll_in_c,
            "UNROLL_OUT_C": self.pe_unroll_embed_dim,
        }

    def data_generate(self):
        torch.manual_seed(0)
        self.patch_embed = QuantizedPatchEmbed(
            img_size=self.in_x,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            in_chans=self.in_c,
            config=self.w_config,
        )
        # get parameters with integer format
        patch_w_1 = q2i(
            self.patch_embed.proj.weight,
            self.w_config["patch_proj"]["weight_width"],
            self.w_config["patch_proj"]["weight_frac_width"],
        )
        print("weight = ", self.patch_embed.proj.weight)
        patch_b_1 = q2i(
            self.patch_embed.proj.bias,
            self.w_config["patch_proj"]["bias_width"],
            self.w_config["patch_proj"]["bias_frac_width"],
        )
        print("patch_b_1 = ", patch_b_1)
        print("bias = ", self.patch_embed.proj.bias)
        self.x = 5 * torch.randn(self.samples, self.in_c, self.in_y, self.in_x)
        self.x_in = q2i(
            self.x,
            self.w_config["patch_proj"]["data_in_width"],
            self.w_config["patch_proj"]["data_in_frac_width"],
        )
        print("x = ", self.x)
        # parameters packs
        self.pe_w_in, self.pe_b_in = self.conv_pack(
            weight=patch_w_1,
            bias=patch_b_1,
            in_channels=self.in_c,
            kernel_size=[self.patch_size, self.patch_size],
            out_channels=self.embed_dim,
            unroll_in_channels=self.pe_unroll_in_c,
            unroll_kernel_out=self.pe_unroll_kernel_out,
            unroll_out_channels=self.pe_unroll_embed_dim,
        )

        self.x_in = self.x_in.permute(0, 2, 3, 1).reshape(-1, self.pe_unroll_in_c)

        self.x_in = self.x_in.flip(0).tolist()

    def sw_compute(self):
        data_out, _ = self.patch_embed(self.x)
        # breakpoint()
        print(data_out)
        output = self.linear_data_pack(
            q2i(data_out, self.data_out_width, self.data_out_frac_width),
            in_y=self.num_patch,
            in_x=self.embed_dim,
            unroll_in_y=1,
            unroll_in_x=self.pe_unroll_embed_dim,
        )
        return output

    def linear_data_pack(self, in_temp, in_y, in_x, unroll_in_y, unroll_in_x):
        ## just what to make a matrix with [np*p][s*d] to tile [np*d][p*s]
        ## assume the in_temp as torch.float
        np = int(in_y / unroll_in_y)
        d = int(in_x / unroll_in_x)
        p = unroll_in_y
        s = unroll_in_x

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

    def conv_pack(
        self,
        weight,
        bias,
        in_channels,
        kernel_size,
        out_channels,
        unroll_in_channels,
        unroll_kernel_out,
        unroll_out_channels,
    ):
        samples = self.samples
        # requires input as a quantized int format
        # weight_pack
        # from (oc,ic/u_ic,u_ic,h,w) to (ic/u_ic,h*w,u_ic,oc)
        reorder_w_tensor = (
            weight.repeat(samples, 1, 1, 1, 1)
            .reshape(
                samples,
                out_channels,
                int(in_channels / unroll_in_channels),
                unroll_in_channels,
                kernel_size[0] * kernel_size[1],
            )
            .permute(0, 2, 4, 3, 1)
        )

        # reverse the final 2 dimension
        # from(samples, int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_kernel_out, int(out_channels/unroll_out_channels), unroll_out_channels)
        # to  (samples, int(out_channels/unroll_out_channels), int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_out_channels, unroll_kernel_out)
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_size[0] * kernel_size[1] * in_channels / unroll_kernel_out),
            unroll_kernel_out,
            int(out_channels / unroll_out_channels),
            unroll_out_channels,
        ).permute(0, 3, 1, 4, 2)

        w_tensor = w_tensor.reshape(
            -1,
            unroll_out_channels * unroll_kernel_out,
        )
        w_in = w_tensor.type(torch.int).flip(0).tolist()
        # bias_pack
        bias_tensor = bias.repeat(samples, 1).reshape(-1, unroll_out_channels)
        b_in = bias_tensor.type(torch.int).flip(0).tolist()
        return w_in, b_in


@cocotb.test()
async def test_fixed_linear(dut):
    # TODO:
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
    dut.weight_valid.value = 0
    dut.bias_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")

    done = False
    cdin = 0
    cpatch_out = 0
    # Set a timeout to avoid deadlock
    for i in range(samples * 400000):
        await FallingEdge(dut.clk)
        # debug_state(dut, "Post-clk")
        dut.rst.value = 0
        dut.bias_valid.value = test_case.patch_embed_bias.pre_compute()
        dut.weight_valid.value = test_case.patch_embed_weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        # start input data
        dut.weight_valid.value, dut.weight.value = test_case.patch_embed_weight.compute(
            dut.weight_ready.value
        )
        dut.bias_valid.value, dut.bias.value = test_case.patch_embed_bias.compute(
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
        # if(dut.data_out_ready.value and dut.data_out_valid.value):
        #     if()
        #     dut.rst.value = 1
        #     dut.data_in_ready.value = 0
        wave_check(dut)
        if dut.data_in_valid.value == 1 and dut.data_in_ready.value == 1:
            cdin += 1
        if dut.data_out_valid.value == 1 and dut.data_out_ready.value == 1:
            cpatch_out += 1
        print("cdin = ", cdin)
        print("cpatch_out = ", cpatch_out)
        # breakpoint()
        if (
            test_case.outputs.is_full()
            and test_case.patch_embed_bias.is_empty()
            and test_case.patch_embed_weight.is_empty()
            and test_case.data_in.is_empty()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave_check:\n\
                {},{} data_in = {}\n\
                {},{} data_out = {}\n\
                ".format(
            dut.conv_inst.fl_instance.data_in_valid.value,
            dut.conv_inst.fl_instance.data_in_ready.value,
            [int(i) for i in dut.conv_inst.fl_instance.data_in.value],
            dut.conv_inst.fl_instance.data_out_valid.value,
            dut.conv_inst.fl_instance.data_out_ready.value,
            [int(i) for i in dut.conv_inst.fl_instance.data_out.value],
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/fixed_pvt.sv",
        "../../../../components/common/cut_data.sv",
        "../../../../components/ViT/fixed_patch_embed.sv",
        "../../../../components/conv/convolution.sv",
        "../../../../components/conv/padding.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/conv/sliding_window.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/split2.sv",
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
        hdl_toplevel="fixed_patch_embed",
        build_args=extra_args,
    )
    runner.test(
        hdl_toplevel="fixed_patch_embed",
        test_module="fixed_patch_embed_tb",
    )


if __name__ == "__main__":
    runner()
