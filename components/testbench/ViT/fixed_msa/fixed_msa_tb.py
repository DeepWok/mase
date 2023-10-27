#!/usr/bin/env python3

import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append("/workspace/components/testbench/ViT/")
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

from pvt_quant import QuantizedAttention
from ha_softmax import generate_table_hardware, generate_table_div_hardware
from z_qlayers import quantize_to_int as q2i

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        # self.seeds = random.randint(0,1000)
        self.data_in_width = 8
        self.data_in_frac_width = 5
        self.weight_q_width = 8
        self.weight_q_frac_width = 4
        self.weight_k_width = 8
        self.weight_k_frac_width = 4
        self.weight_v_width = self.weight_k_width
        self.weight_v_frac_width = self.weight_k_frac_width
        self.weight_p_width = 8
        self.weight_p_frac_width = 4

        self.bias_q_width = 8
        self.bias_q_frac_width = 5
        self.bias_k_width = 8
        self.bias_k_frac_width = 5
        self.bias_v_width = self.bias_k_width
        self.bias_v_frac_width = self.bias_k_frac_width
        self.bias_p_width = 8
        self.bias_p_frac_width = 5

        self.data_q_width = 8
        self.data_q_frac_width = 5
        self.data_k_width = 8
        self.data_k_frac_width = 5
        self.data_v_width = 8
        self.data_v_frac_width = 5
        self.data_s_width = 8
        self.data_s_frac_width = 5
        self.exp_width = 8
        self.exp_frac_width = 5
        self.div_width = 10
        self.data_s_softmax_width = 8
        self.data_s_softmax_frac_width = 4
        self.data_z_width = 8
        self.data_z_frac_width = 3

        self.div_width = 10

        self.w_config = {
            "q_proj": {
                "name": "integer",
                "weight_width": self.weight_q_width,
                "weight_frac_width": self.weight_q_frac_width,
                "data_in_width": self.data_in_width,
                "data_in_frac_width": self.data_in_frac_width,
                "bias_width": self.bias_q_width,
                "bias_frac_width": self.bias_q_frac_width,
            },
            "kv_proj": {
                "name": "integer",
                "weight_width": self.weight_k_width,
                "weight_frac_width": self.weight_k_frac_width,
                "data_in_width": self.data_in_width,
                "data_in_frac_width": self.data_in_frac_width,
                "bias_width": self.bias_k_width,
                "bias_frac_width": self.bias_k_frac_width,
            },
            "z_proj": {
                "name": "integer",
                "weight_width": self.weight_p_width,
                "weight_frac_width": self.weight_p_frac_width,
                "data_in_width": self.data_z_width,
                "data_in_frac_width": self.data_z_frac_width,
                "bias_width": self.bias_p_width,
                "bias_frac_width": self.bias_p_frac_width,
            },
            "softmax": {
                "exp_width": self.exp_width,
                "exp_frac_width": self.exp_frac_width,
                "div_width": self.div_width,
                "data_in_width": self.data_s_width,
                "data_in_frac_width": self.data_s_frac_width,
                "data_out_width": self.data_s_softmax_width,
                "data_out_frac_width": self.data_s_softmax_frac_width,
            },
            "attn_matmul": {
                "name": "integer",
                "data_in_width": self.data_q_width,
                "data_in_frac_width": self.data_q_frac_width,
                "weight_width": self.data_k_width,
                "weight_frac_width": self.data_k_frac_width,
            },
            "z_matmul": {
                "name": "integer",
                "data_in_width": self.data_s_width,
                "data_in_frac_width": self.data_s_frac_width,
                "weight_width": self.data_v_width,
                "weight_frac_width": self.data_v_frac_width,
            },
        }
        self.out_width = 8
        self.out_frac_width = 5

        self.in_y = 8
        self.in_x = 8
        self.unroll_in_x = 2
        self.unroll_w_y = 4
        self.num_heads = 2
        self.w_y = self.in_x
        self.unroll_in_y = 1
        self.wp_y = self.in_x
        self.unroll_wp_y = self.unroll_in_x

        self.in_parallelism = self.unroll_in_y
        self.in_num_parallelism = self.in_y // self.unroll_in_y

        self.in_size = self.unroll_in_x
        self.in_depth = self.in_x // self.unroll_in_x

        # noted num_heads * wqkv_p * wqkv_np should be = in_s * in_d
        self.wqkv_parallelism = self.unroll_w_y
        self.wqkv_num_parallelism = self.w_y // (self.unroll_w_y * self.num_heads)

        assert (
            self.num_heads * self.wqkv_parallelism * self.wqkv_num_parallelism
            == self.in_size * self.in_depth
        ), "should have num_heads * wqkv_p * wqkv_np == in_s * in_d"

        self.wp_parallelism = self.unroll_wp_y
        self.wp_num_parallelism = self.wp_y // self.unroll_wp_y

        assert (
            self.wp_parallelism * self.wp_num_parallelism
            == self.in_size * self.in_depth
        ), "should have wp_p * wp_np == in_s * in_d"

        self.wp_size = self.num_heads * self.wqkv_parallelism
        self.wp_depth = self.wqkv_num_parallelism
        # data_generate
        self.data_generate()
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
            "IN_WIDTH": self.data_in_width,
            "IN_FRAC_WIDTH": self.data_in_frac_width,
            "WQ_WIDTH": self.weight_q_width,
            "WQ_FRAC_WIDTH": self.weight_q_frac_width,
            "WK_WIDTH": self.weight_k_width,
            "WK_FRAC_WIDTH": self.weight_k_frac_width,
            "WV_WIDTH": self.weight_v_width,
            "WV_FRAC_WIDTH": self.weight_v_frac_width,
            "WP_WIDTH": self.weight_p_width,
            "WP_FRAC_WIDTH": self.weight_p_frac_width,
            "BQ_WIDTH": self.bias_q_width,
            "BQ_FRAC_WIDTH": self.bias_q_frac_width,
            "BK_WIDTH": self.bias_k_width,
            "BK_FRAC_WIDTH": self.bias_k_frac_width,
            "BV_WIDTH": self.bias_v_width,
            "BV_FRAC_WIDTH": self.bias_v_frac_width,
            "BP_WIDTH": self.bias_p_width,
            "BP_FRAC_WIDTH": self.bias_p_frac_width,
            "DQ_WIDTH": self.data_q_width,
            "DQ_FRAC_WIDTH": self.data_q_frac_width,
            "DK_WIDTH": self.data_k_width,
            "DK_FRAC_WIDTH": self.data_k_frac_width,
            "DV_WIDTH": self.data_v_width,
            "DV_FRAC_WIDTH": self.data_v_frac_width,
            "DS_WIDTH": self.data_s_width,
            "DS_FRAC_WIDTH": self.data_s_frac_width,
            "EXP_WIDTH": self.w_config["softmax"]["exp_width"],
            "EXP_FRAC_WIDTH": self.w_config["softmax"]["exp_frac_width"],
            "DIV_WIDTH": self.w_config["softmax"]["div_width"],
            "DS_SOFTMAX_WIDTH": self.w_config["softmax"]["data_out_width"],
            "DS_SOFTMAX_FRAC_WIDTH": self.w_config["softmax"]["data_out_frac_width"],
            "DZ_WIDTH": self.data_z_width,
            "DZ_FRAC_WIDTH": self.data_z_frac_width,
            "OUT_WIDTH": self.out_width,
            "OUT_FRAC_WIDTH": self.out_frac_width,
            "UNROLL_IN_Y": self.in_parallelism,
            "IN_Y": self.in_num_parallelism * self.in_parallelism,
            "UNROLL_IN_X": self.in_size,
            "IN_X": self.in_depth * self.in_size,
            "NUM_HEADS": self.num_heads,
            "UNROLL_WQKV_Y": self.wqkv_parallelism,
            "WQKV_Y": self.wqkv_parallelism * self.wqkv_num_parallelism,
            "UNROLL_WP_Y": self.wp_parallelism,
            "WP_Y": self.wp_parallelism * self.wp_num_parallelism,
        }

    def data_generate(self):
        # generate data
        samples = self.samples
        torch.manual_seed(2)
        # breakpoint()
        self.x = torch.randn((samples, self.in_y, self.in_x))
        self.att = QuantizedAttention(
            dim=self.in_x,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            config=self.w_config,
        )
        input_tensor = q2i(self.x, self.data_in_width, self.data_in_frac_width)
        att_wq = q2i(self.att.q.weight, self.weight_q_width, self.weight_q_frac_width)
        att_wkv = q2i(self.att.kv.weight, self.weight_k_width, self.weight_k_frac_width)
        wqkv_tensor = torch.cat((att_wq, att_wkv), 0)
        wqkv_tensor = wqkv_tensor.reshape(3, self.in_x, self.in_x)
        wqkv_tensor = wqkv_tensor.reshape(self.in_x * 3, self.in_x).repeat(
            samples, 1, 1
        )

        att_bq = q2i(self.att.q.bias, self.bias_q_width, self.bias_q_frac_width)
        att_bkv = q2i(self.att.kv.bias, self.bias_k_width, self.bias_k_frac_width)
        bqkv_tensor = torch.cat((att_bq, att_bkv), 0)
        bqkv_tensor = bqkv_tensor.reshape(3, self.in_x)
        bqkv_tensor = bqkv_tensor.reshape(-1).repeat(samples, 1)

        wp_tensor = q2i(
            self.att.proj.weight, self.weight_p_width, self.weight_p_frac_width
        ).repeat(samples, 1, 1)
        bp_tensor = q2i(
            self.att.proj.bias, self.bias_p_width, self.bias_p_frac_width
        ).repeat(samples, 1)

        logger.debug(
            "input data: \n\
        d_tensor = \n{}\n\
        wqkv_tensor = \n{}\n\
        bqkv_tensor = \n{}\n\
        wp_tensor = \n{}\n\
        bp_tensor = \n{}\n\
        ".format(
                input_tensor, wqkv_tensor, bqkv_tensor, wp_tensor, bp_tensor
            )
        )
        # generate hash table
        exp_table = generate_table_hardware(
            self.att.scale,
            self.w_config["softmax"]["data_in_width"],
            self.w_config["softmax"]["data_in_frac_width"],
            self.w_config["softmax"]["exp_width"],
            self.w_config["softmax"]["exp_frac_width"],
        ).tolist()
        div_table = generate_table_div_hardware(
            self.w_config["softmax"]["div_width"],
            self.w_config["softmax"]["data_out_width"],
            self.w_config["softmax"]["data_out_frac_width"],
        ).tolist()
        with open(r"exp_init.mem", "w") as fp:
            for item in exp_table:
                # write each item on a new lineformat(addr[i] ,f'0{width}b'
                fp.write(
                    "%s\n"
                    % format(item, f'0{self.w_config["softmax"]["exp_width"]//4}x')
                )
        with open(r"div_init.mem", "w") as fp:
            for item in div_table:
                # write each item on a new line
                fp.write(
                    "%s\n"
                    % format(item, f'0{self.w_config["softmax"]["data_out_width"]//4}x')
                )
        # data_pack
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

        wqkv = wqkv_tensor.reshape(
            samples, 3, num_heads, wqkv_num_parallelism, wqkv_parallelism, dim
        ).permute(1, 0, 3, 2, 4, 5)
        wqkv = wqkv.reshape(3, samples, dim, dim)
        bqkv = bqkv_tensor.reshape(
            samples, 3, num_heads, wqkv_num_parallelism, wqkv_parallelism
        ).permute(1, 0, 3, 2, 4)
        bqkv = bqkv.reshape(3, samples, dim)

        wp = wp_tensor.reshape(
            samples * dim, num_heads, wqkv_num_parallelism, wqkv_parallelism
        )
        wp = wp.permute(0, 2, 1, 3).reshape(samples, dim, dim)

        wq = wqkv[0]
        wk = wqkv[1]
        wv = wqkv[2]

        bq = bqkv[0]
        bk = bqkv[1]
        bv = bqkv[2]
        self.data_in = self.data_pack(
            input_tensor, in_num_parallelism, in_depth, in_parallelism, in_size
        )
        self.wq_in = self.data_pack(
            wq, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        self.wk_in = self.data_pack(
            wk, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        self.wv_in = self.data_pack(
            wv, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        self.wp_in = self.data_pack(
            wp,
            wp_num_parallelism,
            wqkv_num_parallelism,
            wp_parallelism,
            num_heads * wqkv_parallelism,
        )

        self.bq_in = self.data_pack(
            bq, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        self.bk_in = self.data_pack(
            bk, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        self.bv_in = self.data_pack(
            bv, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        self.bp_in = self.data_pack(bp_tensor, 1, wp_num_parallelism, 1, wp_parallelism)

        self.data_in.reverse()
        self.wq_in.reverse()
        self.wk_in.reverse()
        self.wv_in.reverse()
        self.wp_in.reverse()
        self.bq_in.reverse()
        self.bk_in.reverse()
        self.bv_in.reverse()
        self.bp_in.reverse()

        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.in_num_parallelism,
            data_specify=self.data_in,
            debug=debug,
        )
        self.weight_q = RandomSource(
            name="weight_q",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=self.wq_in,
            debug=debug,
        )
        self.weight_k = RandomSource(
            name="weight_k",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=self.wk_in,
            debug=debug,
        )
        self.weight_v = RandomSource(
            name="weight_v",
            samples=samples * self.in_depth * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.wqkv_num_parallelism,
            data_specify=self.wv_in,
            debug=debug,
        )
        self.weight_p = RandomSource(
            name="weight_p",
            samples=samples * self.wp_depth * self.wp_num_parallelism,
            num=self.wp_parallelism * self.wp_size,
            max_stalls=2 * samples * self.wp_depth * self.wp_num_parallelism,
            data_specify=self.wp_in,
            debug=debug,
        )
        self.bias_q = RandomSource(
            name="bias_q",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=self.bq_in,
            debug=debug,
        )
        self.bias_k = RandomSource(
            name="bias_k",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=self.bk_in,
            debug=debug,
        )
        self.bias_v = RandomSource(
            name="bias_v",
            samples=samples * self.wqkv_num_parallelism,
            num=self.num_heads * self.wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=self.bv_in,
            debug=debug,
        )
        self.bias_p = RandomSource(
            name="bias_p",
            samples=samples * self.wp_num_parallelism,
            num=self.wp_parallelism,
            max_stalls=2 * samples,
            data_specify=self.bp_in,
            debug=debug,
        )

    def sw_compute(self):
        # get the matrix out result
        # from M[num_parallelism][depth],
        # and the element in M is m[parallelism][size]
        # to M_out[in1_num_parallelism][in2_num_parallelism]
        # the element in M_out is m_out[in1_parallelism][in2_parallelism]

        # collect all the input
        # calculate the output
        # cut the output to smaller sets
        data_out = self.att(self.x)
        output = self.data_pack(
            q2i(data_out, self.out_width, self.out_frac_width),
            self.in_num_parallelism,
            self.wp_num_parallelism,
            self.in_parallelism,
            self.wp_parallelism,
        )
        return output

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
async def test_msa(dut):
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
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")
    done = False
    # Set a timeout to avoid deadlock
    cdin = 0
    cdata_out = 0
    for i in range(samples * 15000):
        await FallingEdge(dut.clk)
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
        # debug_state(dut, "in compute")
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
        if dut.data_in_valid.value == 1 and dut.data_in_ready.value == 1:
            cdin += 1
        if dut.data_out_valid.value == 1 and dut.data_out_ready.value == 1:
            cdata_out += 1
        print("cdin = ", cdin)
        print("cdata_out = ", cdata_out)
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
            {},{},data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
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
        "../../../../components/ViT/fixed_msa.sv",
        "../../../../components/ViT/hash_softmax.sv",
        "../../../../components/attention/fixed_self_att.sv",
        "../../../../components/attention/fixed_att.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/split2.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/cast/fixed_rounding.sv",
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
    # for i in range(200):
    runner()
