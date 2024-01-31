#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging, sys

from mase_cocotb.random_test import RandomSource, RandomSink, check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from einops import rearrange
import torch
from mase_cocotb.z_qlayers import quantize_to_int as q2i
from chop.models.manual.quant_utils import get_quantized_func

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 8
        self.data_in_frac_width = 7
        self.weight_width = 8
        self.weight_frac_width = 5
        self.data_out_width = 8
        self.data_out_frac_width = 5

        self.has_bias = 0
        self.bias_width = 6
        self.bias_frac_width = 1

        self.in_parallelism = 1
        self.in_num_parallelism = 2

        self.in_size = 1
        self.weight_size = self.in_size

        self.w_parallelism = 2
        self.w_num_parallelism = 2
        self.weight_columns = self.w_parallelism * self.w_num_parallelism
        self.in_depth = 2

        self.b_size = self.w_parallelism
        self.b_depth = self.w_num_parallelism

        _, _, _, d, w, b = self.data_generate()
        self.bias = RandomSource(
            name="data_in",
            samples=samples * self.in_num_parallelism * self.b_depth,
            num=self.in_parallelism * self.b_size,
            max_stalls=2 * samples,
            data_specify=b,
            debug=debug,
        )
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples,
            data_specify=d,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.weight_size,
            max_stalls=2 * samples,
            data_specify=w,
            debug=debug,
        )

        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.w_num_parallelism,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN1_WIDTH": self.data_in_width,
            "IN1_FRAC_WIDTH": self.data_in_frac_width,
            "IN2_WIDTH": self.weight_width,
            "IN2_FRAC_WIDTH": self.weight_frac_width,
            "HAS_BIAS": self.has_bias,
            "BIAS_WIDTH": self.bias_width,
            "BIAS_FRAC_WIDTH": self.bias_frac_width,
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN1_Y": self.in_parallelism * self.in_num_parallelism,
            "UNROLL_IN1_Y": self.in_parallelism,
            "IN1_X": self.in_size * self.in_depth,
            "UNROLL_IN1_X": self.in_size,
            "IN2_Y": self.w_parallelism * self.w_num_parallelism,
            "UNROLL_IN2_Y": self.w_parallelism,
        }

    def data_generate(self):
        torch.manual_seed(6798317339854970458)
        # print("seed = ", torch.seed(6798317339854970458))
        # breakpoint()
        in_features = self.in_size * self.in_depth
        out_features = self.w_parallelism * self.w_num_parallelism
        in_dim = self.in_parallelism * self.in_num_parallelism

        bias_tensor = 5 * torch.randn((self.samples, in_dim, out_features))
        weight_tensor = 5 * torch.randn((self.samples, out_features, in_features))
        data_tensor = 5 * torch.randn((self.samples, in_dim, in_features))

        data_in = self.data_pack(
            q2i(data_tensor, self.data_in_width, self.data_in_frac_width),
            np=self.in_num_parallelism,
            d=self.in_depth,
            p=self.in_parallelism,
            s=self.in_size,
        )

        weight_in = self.data_pack(
            q2i(weight_tensor, self.weight_width, self.weight_frac_width),
            np=self.w_num_parallelism,
            d=self.in_depth,
            p=self.w_parallelism,
            s=self.in_size,
        )

        bias_in = self.data_pack(
            q2i(bias_tensor, self.bias_width, self.bias_frac_width),
            np=self.in_num_parallelism,
            d=self.w_num_parallelism,
            p=self.in_parallelism,
            s=self.w_parallelism,
        )
        logger.debug(
            "\n\
        data_tensor = {} \n\
        weight_tensor = {} \n\
        data_in = {} \n\
        weight_in = {} ".format(
                data_tensor,
                weight_tensor,
                data_in,
                weight_in,
            )
        )
        data_in.reverse()
        weight_in.reverse()
        bias_in.reverse()
        # NOTE: weight in should transpose here
        return (
            data_tensor,
            weight_tensor.transpose(-2, -1),
            bias_tensor,
            data_in,
            weight_in,
            bias_in,
        )

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
        data_in, weight_in, _, _, _, _ = self.data_generate()
        in_features = self.in_size * self.in_depth
        out_features = self.w_parallelism * self.w_num_parallelism
        in_dim = data_in.shape[1]
        config = {
            "name": "integer",
            "weight_width": self.weight_width,
            "weight_frac_width": self.weight_frac_width,
            "data_in_width": self.data_in_width,
            "data_in_frac_width": self.data_in_frac_width,
            "bias_width": self.bias_width,
            "bias_frac_width": self.bias_frac_width,
        }
        matmul = get_quantized_func("matmul", config)
        data_out = matmul(data_in, weight_in, config)
        print(data_out)
        data_out = q2i(data_out, self.data_out_width, self.data_out_frac_width)
        output = self.data_pack(
            data_out,
            np=self.in_num_parallelism,
            d=self.w_num_parallelism,
            p=self.in_parallelism,
            s=self.w_parallelism,
        )
        return output


# Check if an is_impossible state is reached
def is_impossible_state(
    data_in2_ready,
    data_in2_valid,
    data_in1_ready,
    data_in1_valid,
    data_out_ready,
    data_out_valid,
):
    return False


def debug_state(dut, state):
    logger.debug(
        "{} State: (data_in2_ready,data_in2_valid,data_in1_ready,data_in1_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.data_in2_ready.value,
            dut.data_in2_valid.value,
            dut.data_in1_ready.value,
            dut.data_in1_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 100
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
    dut.data_in2_valid.value = 0
    dut.data_in1_valid.value = 0
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
        dut.data_in2_valid.value = test_case.weight.pre_compute()
        dut.data_in1_valid.value = test_case.data_in.pre_compute()
        dut.bias_valid.value = test_case.bias.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        # start input data
        #
        dut.data_in2_valid.value, dut.data_in2.value = test_case.weight.compute(
            dut.data_in2_ready.value
        )
        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
        dut.data_in1_valid.value, dut.data_in1.value = test_case.data_in.compute(
            dut.data_in1_ready.value
        )

        await Timer(1, units="ns")
        # breakpoint()

        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        logger.debug(
            "wave_check:\n\
            {},{} ib_data_in = {}\n\
            {},{} ib_weight = {}\n\
            {},{} data_out = {}\n\
            {},{} cast_data = {}\n\
            ".format(
                dut.ib_data_in_valid.value,
                dut.ib_data_in_ready.value,
                [int(i) for i in dut.ib_data_in.value],
                dut.ib_weight_valid.value,
                dut.ib_weight_ready.value,
                [int(i) for i in dut.ib_weight.value],
                dut.data_out_valid.value,
                dut.data_out_ready.value,
                [int(i) for i in dut.data_out.value],
                dut.inst_fmmc.data_out_valid.value,
                dut.inst_fmmc.data_out_ready.value,
                [int(i) for i in dut.inst_fmmc.cast_data.value],
            )
        )
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
        "../../../mase_components/matmul/rtl/fixed_matmul.sv",
        "../../../mase_components/common/rtl/input_buffer.sv",
        "../../../mase_components/common/rtl/blk_mem_gen_0.sv",
        "../../../mase_components/common/rtl/skid_buffer.sv",
        "../../../mase_components/common/rtl/unpacked_skid_buffer.sv",
        "../../../mase_components/common/rtl/join2.sv",
        "../../../mase_components/linear/rtl/fixed_linear.sv",
        "../../../mase_components/cast/rtl/fixed_round.sv",
        "../../../mase_components/cast/rtl/fixed_rounding.sv",
        "../../../mase_components/cast/rtl/fixed_cast.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_matmul_core.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_dot_product.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_accumulator.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_vector_mult.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_adder_tree.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_adder_tree_layer.sv",
        "../../../mase_components/fixed_arith/rtl/fixed_mult.sv",
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
        hdl_toplevel="fixed_matmul",
        build_args=extra_args,
    )
    runner.test(
        hdl_toplevel="fixed_matmul",
        test_module="fixed_matmul_tb",
    )


if __name__ == "__main__":
    runner()
