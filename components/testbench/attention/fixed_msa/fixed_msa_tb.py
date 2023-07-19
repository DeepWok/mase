#!/usr/bin/env python3

import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random, os, math, logging, sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# from torchsummary import summary
from einops import rearrange, reduce, repeat


from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from QAttention import QHAttention

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 32
        self.data_in_frac_width = 1
        self.weight_width = 16
        self.weight_frac_width = 2
        self.bias_width = 16
        self.bias_frac_width = 2

        self.in_parallelism = 2
        self.in_num_parallelism = 1

        self.in_size = 3
        self.in_depth = 2

        # noted num_heads * wqkv_p * wqkv_np should be = in_s * in_d
        self.num_heads = 2
        self.wqkv_parallelism = 1
        self.wqkv_num_parallelism = 3

        assert (
            self.num_heads * self.wqkv_parallelism * self.wqkv_num_parallelism
            == self.in_size * self.in_depth
        ), "should have num_heads * wqkv_p * wqkv_np == in_s * in_d"

        self.wp_parallelism = 3
        self.wp_num_parallelism = 2

        assert (
            self.wp_parallelism * self.wp_num_parallelism
            == self.in_size * self.in_depth
        ), "should have wp_p * wp_np == in_s * in_d"

        self.wp_size = self.num_heads * self.wqkv_parallelism
        self.wp_depth = self.wqkv_num_parallelism
        # data_generate
        (
            _,
            _,
            _,
            _,
            _,
            test_in,
            test_wq,
            test_wk,
            test_wv,
            test_wp,
            test_bq,
            test_bk,
            test_bv,
            test_bp,
        ) = self.data_generate()
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.in_num_parallelism,
            data_specify=test_in,
            debug=debug,
        )
        self.weight_q = RandomSource(
            name="weight_q",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=test_wq,
            debug=debug,
        )
        self.weight_k = RandomSource(
            name="weight_k",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=test_wk,
            debug=debug,
        )
        self.weight_v = RandomSource(
            name="weight_v",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=test_wv,
            debug=debug,
        )
        self.weight_p = RandomSource(
            name="weight_p",
            samples=samples * self.wp_depth * self.wp_num_parallelism,
            num=self.wp_parallelism * self.wp_size,
            max_stalls=2 * samples * self.wp_depth * self.wp_num_parallelism,
            data_specify=test_wp,
            debug=debug,
        )
        self.bias_q = RandomSource(
            name="bias_q",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bq,
            debug=debug,
        )
        self.bias_k = RandomSource(
            name="bias_k",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bk,
            debug=debug,
        )
        self.bias_v = RandomSource(
            name="bias_v",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bv,
            debug=debug,
        )
        self.bias_p = RandomSource(
            name="bias_p",
            samples=samples * self.wp_num_parallelism,
            num=self.wp_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bp,
            debug=debug,
        )
        ## remain modification
        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.wp_num_parallelism,
            max_stalls=2 * samples * self.in_num_parallelism * self.wp_parallelism,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_in_width,
            "DATA_FRAC_WIDTH": self.data_in_frac_width,
            "WEIGHT_WIDTH": self.weight_width,
            "W_FRAC_WIDTH": self.weight_frac_width,
            "BIAS_WIDTH": self.bias_width,
            "BIAS_FRAC_WIDTH": self.bias_frac_width,
            "IN_PARALLELISM": self.in_parallelism,
            "IN_NUM_PARALLELISM": self.in_num_parallelism,
            "IN_SIZE": self.in_size,
            "IN_DEPTH": self.in_depth,
            "NUM_HEADS": self.num_heads,
            "WQKV_PARALLELISM": self.wqkv_parallelism,
            "WQKV_NUM_PARALLELISM": self.wqkv_num_parallelism,
            "WP_PARALLELISM": self.wp_parallelism,
            "WP_NUM_PARALLELISM": self.wp_num_parallelism,
        }

    def sw_compute(self):
        # get the matrix out result
        # from M[num_parallelism][depth],
        # and the element in M is m[parallelism][size]
        # to M_out[in1_num_parallelism][in2_num_parallelism]
        # the element in M_out is m_out[in1_parallelism][in2_parallelism]

        # collect all the input
        # breakpoint()
        (
            d_tensor,
            wqkv_tensor,
            bqkv_tensor,
            wp_tensor,
            bp_tensor,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.data_generate()
        logger.debug(
            "input data: \n\
        d_tensor = \n{}\n\
        wqkv_tensor = \n{}\n\
        bqkv_tensor = \n{}\n\
        wp_tensor = \n{}\n\
        bp_tensor = \n{}\n\
        ".format(
                d_tensor, wqkv_tensor, bqkv_tensor, wp_tensor, bp_tensor
            )
        )
        # calculate the output
        # cut the output to smaller sets
        ref = []
        output = torch.zeros(
            (
                self.samples,
                self.in_num_parallelism * self.in_parallelism,
                self.wp_num_parallelism * self.wp_parallelism,
            )
        )
        for i in range(self.samples):
            qatt = QHAttention(
                d_tensor[i].shape[2],
                self.num_heads,
                wqkv_tensor[i],
                wp_tensor[i],
                bqkv_tensor[i],
                bp_tensor[i],
                self.data_in_width,
                self.data_in_frac_width,
                self.weight_width,
                self.weight_frac_width,
                self.bias_width,
                self.bias_frac_width,
            )
            # calculate
            out_temp = rearrange(qatt(d_tensor[i]), "b r c->(b r) c ", b=1)
            output[i] = out_temp

        out_data = self.data_pack(
            output,
            self.in_num_parallelism,
            self.wp_num_parallelism,
            self.in_parallelism,
            self.wp_parallelism,
        )
        return out_data

    def data_generate(self):
        samples = self.samples
        in_num_parallelism = self.in_num_parallelism
        in_depth = self.in_depth
        in_parallelism = self.in_parallelism
        in_size = self.in_size
        wqkv_parallelism = self.wqkv_parallelism
        wqkv_num_parallelism = self.wqkv_num_parallelism
        num_heads = self.num_heads
        wp_parallelism = self.wp_parallelism
        wp_num_parallelism = self.wp_num_parallelism

        B = 1
        N = in_parallelism * in_num_parallelism
        dim = in_size * in_depth

        torch.manual_seed(0)
        wqkv_tensor = torch.randint(5, (samples, dim * 3, dim), dtype=float)
        wqkv = wqkv_tensor.reshape(
            samples, num_heads, 3, wqkv_num_parallelism, wqkv_parallelism, dim
        ).permute(2, 0, 3, 1, 4, 5)
        wqkv = wqkv.reshape(3, samples, dim, dim)
        bqkv_tensor = torch.randint(5, (samples, dim * 3), dtype=float)
        bqkv = bqkv_tensor.reshape(
            samples, num_heads, 3, wqkv_num_parallelism, wqkv_parallelism
        ).permute(2, 0, 3, 1, 4)
        bqkv = bqkv.reshape(3, samples, dim)

        wp_tensor = torch.randint(5, (samples, dim, dim), dtype=float)
        wp = wp_tensor.reshape(
            samples * dim, num_heads, wqkv_num_parallelism, wqkv_parallelism
        )
        wp = wp.permute(0, 2, 1, 3).reshape(samples, dim, dim)
        bp_tensor = torch.randint(5, (samples, dim), dtype=float)

        input_tensor = torch.randint(5, (samples, B, N, dim), dtype=float)
        wq = wqkv[0]
        wk = wqkv[1]
        wv = wqkv[2]

        bq = bqkv[0]
        bk = bqkv[1]
        bv = bqkv[2]
        data_in = self.data_pack(
            input_tensor, in_num_parallelism, in_depth, in_parallelism, in_size
        )
        wq_in = self.data_pack(
            wq, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wk_in = self.data_pack(
            wk, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wv_in = self.data_pack(
            wv, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wp_in = self.data_pack(
            wp,
            wp_num_parallelism,
            wqkv_num_parallelism,
            wp_parallelism,
            num_heads * wqkv_parallelism,
        )

        bq_in = self.data_pack(
            bq, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bk_in = self.data_pack(
            bk, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bv_in = self.data_pack(
            bv, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bp_in = self.data_pack(bp_tensor, 1, wp_num_parallelism, 1, wp_parallelism)

        data_in.reverse()
        wq_in.reverse()
        wk_in.reverse()
        wv_in.reverse()
        wp_in.reverse()
        bq_in.reverse()
        bk_in.reverse()
        bv_in.reverse()
        bp_in.reverse()
        return (
            input_tensor,
            wqkv_tensor,
            bqkv_tensor,
            wp_tensor,
            bp_tensor,
            data_in,
            wq_in,
            wk_in,
            wv_in,
            wp_in,
            bq_in,
            bk_in,
            bv_in,
            bp_in,
        )

    def data_pack(self, in_temp, np, d, p, s):
        # assum in_temp.shape = (samples, batch = 1, N,dim)
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


def debug_state(dut, state):
    logger.debug(
        "{} State: (wq_ready,wq_valid,wk_ready,wk_valid,wv_ready,wv_valid,in_ready,in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{},{},{},{},{})".format(
            state,
            dut.weight_q_ready.value,
            dut.weight_q_valid.value,
            dut.weight_k_ready.value,
            dut.weight_k_valid.value,
            dut.weight_v_ready.value,
            dut.weight_v_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_att(dut):
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
    dut.weight_q_valid.value = 0
    dut.weight_k_valid.value = 0
    dut.weight_v_valid.value = 0
    dut.weight_p_valid.value = 0
    dut.bias_q_valid.value = 0
    dut.bias_k_valid.value = 0
    dut.bias_v_valid.value = 0
    dut.bias_p_valid.value = 0
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
    for i in range(samples * 150):
        await FallingEdge(dut.clk)
        # breakpoint()
        dut.weight_q_valid.value = test_case.weight_q.pre_compute()
        dut.weight_k_valid.value = test_case.weight_k.pre_compute()
        dut.weight_v_valid.value = test_case.weight_v.pre_compute()
        dut.weight_p_valid.value = test_case.weight_p.pre_compute()
        dut.bias_q_valid.value = test_case.bias_q.pre_compute()
        dut.bias_k_valid.value = test_case.bias_k.pre_compute()
        dut.bias_v_valid.value = test_case.bias_v.pre_compute()
        dut.bias_p_valid.value = test_case.bias_p.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "in compute")
        dut.weight_q_valid.value, dut.weight_q.value = test_case.weight_q.compute(
            dut.weight_q_ready.value
        )
        dut.weight_k_valid.value, dut.weight_k.value = test_case.weight_k.compute(
            dut.weight_k_ready.value
        )
        dut.weight_v_valid.value, dut.weight_v.value = test_case.weight_v.compute(
            dut.weight_v_ready.value
        )
        dut.weight_p_valid.value, dut.weight_p.value = test_case.weight_p.compute(
            dut.weight_p_ready.value
        )

        dut.bias_q_valid.value, dut.bias_q.value = test_case.bias_q.compute(
            dut.bias_q_ready.value
        )
        dut.bias_k_valid.value, dut.bias_k.value = test_case.bias_k.compute(
            dut.bias_k_ready.value
        )
        dut.bias_v_valid.value, dut.bias_v.value = test_case.bias_v.compute(
            dut.bias_v_ready.value
        )
        dut.bias_p_valid.value, dut.bias_p.value = test_case.bias_p.compute(
            dut.bias_p_ready.value
        )

        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        wave_check(dut)
        if (
            test_case.weight_q.is_empty()
            and test_case.weight_k.is_empty()
            and test_case.weight_v.is_empty()
            and test_case.weight_p.is_empty()
            and test_case.bias_q.is_empty()
            and test_case.bias_k.is_empty()
            and test_case.bias_v.is_empty()
            and test_case.bias_p.is_empty()
            and test_case.data_in.is_empty()
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
            {},{},weight_q = {} \n\
            {},{},weight_k = {} \n\
            {},{},weight_v = {} \n\
            {},{},weight_p = {} \n\
            {},{},bias_q = {} \n\
            {},{},bias_k = {} \n\
            {},{},bias_v = {} \n\
            {},{},bias_p = {} \n\
            {},{},data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.weight_q_valid.value,
            dut.weight_q_ready.value,
            [int(i) for i in dut.weight_q.value],
            dut.weight_k_valid.value,
            dut.weight_k_ready.value,
            [int(i) for i in dut.weight_k.value],
            dut.weight_v_valid.value,
            dut.weight_v_ready.value,
            [int(i) for i in dut.weight_v.value],
            dut.weight_p_valid.value,
            dut.weight_p_ready.value,
            [int(i) for i in dut.weight_p.value],
            dut.bias_q_valid.value,
            dut.bias_q_ready.value,
            [int(i) for i in dut.bias_q.value],
            dut.bias_k_valid.value,
            dut.bias_k_ready.value,
            [int(i) for i in dut.bias_k.value],
            dut.bias_v_valid.value,
            dut.bias_v_ready.value,
            [int(i) for i in dut.bias_v.value],
            dut.bias_p_valid.value,
            dut.bias_p_ready.value,
            [int(i) for i in dut.bias_p.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )

    logger.debug(
        "wave of sa_out:\n\
            {},{},sa_out = {} \n\
            ".format(
            dut.sa_out_valid.value,
            dut.sa_out_ready.value,
            [int(i) for i in dut.sa_out.value],
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/attention/fixed_msa.sv",
        "../../../../components/attention/fixed_self_att.sv",
        "../../../../components/attention/fixed_att.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/ram_block.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
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
        hdl_toplevel="fixed_msa",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="fixed_msa", test_module="fixed_msa_tb")


if __name__ == "__main__":
    runner()
