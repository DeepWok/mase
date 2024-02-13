#!/usr/bin/env python3

import os, logging
import torch

# from torchsummary import summary
from einops import rearrange

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

from components.ViT.test.helpers.ha_softmax import (
    generate_table_hardware,
    generate_table_div_hardware,
)
from components.ViT.test.helpers.pvt_quant import QuantizedAttention

from mase_cocotb.runner import mase_runner
from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.z_qlayers import quantize_to_int as q2i

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        # width config
        self.samples = samples
        self.data_in_width = 8
        self.data_in_frac_width = 5
        self.weight_q_width = 6
        self.weight_q_frac_width = 6
        self.weight_k_width = 6
        self.weight_k_frac_width = 6
        self.weight_v_width = 6
        self.weight_v_frac_width = 6

        self.bias_q_width = 6
        self.bias_q_frac_width = 6
        self.bias_k_width = 6
        self.bias_k_frac_width = 6
        self.bias_v_width = 6
        self.bias_v_frac_width = 6

        self.data_q_width = 8
        self.data_q_frac_width = 6
        self.data_k_width = 8
        self.data_k_frac_width = 6
        self.data_v_width = 8
        self.data_v_frac_width = 6
        self.data_s_width = 8
        self.data_s_frac_width = 6
        self.exp_width = 8
        self.exp_frac_width = 5
        self.div_width = 10
        self.data_s_softmax_width = 8
        self.data_s_softmax_width = 7
        self.data_z_width = 8
        self.data_z_frac_width = 6
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
            "attn_matmul": {
                "name": "integer",
                "data_in_width": self.data_q_width,
                "data_in_frac_width": self.data_q_frac_width,
                "weight_width": self.data_k_width,
                "weight_frac_width": self.data_k_frac_width,
            },
            "z_matmul": {
                "name": "integer",
                "data_in_width": self.data_s_softmax_width,
                "data_in_frac_width": self.data_s_softmax_width,
                "weight_width": self.data_v_width,
                "weight_frac_width": self.data_v_frac_width,
            },
            "softmax": {
                "exp_width": self.exp_width,
                "exp_frac_width": self.exp_frac_width,
                "div_width": self.div_width,
                "data_in_width": self.data_s_width,
                "data_in_frac_width": self.data_s_frac_width,
                "data_out_width": self.data_s_softmax_width,
                "data_out_frac_width": self.data_s_softmax_width,
            },
        }

        self.in_parallelism = 1
        self.in_num_parallelism = 2

        self.in_size = 4
        self.in_depth = 2

        self.w_parallelism = 4
        self.w_num_parallelism = 2
        (
            test_in,
            test_wq,
            test_wk,
            test_wv,
            test_bq,
            test_bk,
            test_bv,
        ) = self.att_data_generate()
        self.soft_max_data_generate(self.att.scale)
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
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            data_specify=test_wq,
            debug=debug,
        )
        self.weight_k = RandomSource(
            name="weight_k",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            data_specify=test_wk,
            debug=debug,
        )
        self.weight_v = RandomSource(
            name="weight_v",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            data_specify=test_wv,
            debug=debug,
        )
        self.bias_q = RandomSource(
            name="bias_q",
            samples=samples * self.w_num_parallelism,
            num=self.w_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bq,
            debug=debug,
        )
        self.bias_k = RandomSource(
            name="bias_k",
            samples=samples * self.w_num_parallelism,
            num=self.w_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bk,
            debug=debug,
        )
        self.bias_v = RandomSource(
            name="bias_v",
            samples=samples * self.w_num_parallelism,
            num=self.w_parallelism,
            max_stalls=2 * samples,
            data_specify=test_bv,
            debug=debug,
        )

        ## remain modification
        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.w_num_parallelism,
            max_stalls=2 * samples * self.in_num_parallelism * self.w_num_parallelism,
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_in_width,
            "DATA_FRAC_WIDTH": self.data_in_frac_width,
            "WQ_WIDTH": self.weight_q_width,
            "WQ_FRAC_WIDTH": self.weight_q_frac_width,
            "WK_WIDTH": self.weight_k_width,
            "WK_FRAC_WIDTH": self.weight_k_frac_width,
            "WV_WIDTH": self.weight_v_width,
            "WV_FRAC_WIDTH": self.weight_v_frac_width,
            "BQ_WIDTH": self.bias_q_width,
            "BQ_FRAC_WIDTH": self.bias_q_frac_width,
            "BK_WIDTH": self.bias_k_width,
            "BK_FRAC_WIDTH": self.bias_k_frac_width,
            "BV_WIDTH": self.bias_v_width,
            "BV_FRAC_WIDTH": self.bias_v_frac_width,
            "DQ_WIDTH": self.data_q_width,
            "DQ_FRAC_WIDTH": self.data_q_frac_width,
            "DK_WIDTH": self.data_k_width,
            "DK_FRAC_WIDTH": self.data_k_frac_width,
            "DV_WIDTH": self.data_v_width,
            "DV_FRAC_WIDTH": self.data_v_frac_width,
            "DS_WIDTH": self.w_config["softmax"]["data_in_width"],
            "DS_FRAC_WIDTH": self.w_config["softmax"]["data_in_frac_width"],
            "EXP_WIDTH": self.w_config["softmax"]["exp_width"],
            "EXP_FRAC_WIDTH": self.w_config["softmax"]["exp_frac_width"],
            "DIV_WIDTH": self.w_config["softmax"]["div_width"],
            "DS_SOFTMAX_WIDTH": self.w_config["softmax"]["data_out_width"],
            "DS_SOFTMAX_FRAC_WIDTH": self.w_config["softmax"]["data_out_frac_width"],
            "DZ_WIDTH": self.data_z_width,
            "DZ_FRAC_WIDTH": self.data_z_frac_width,
            "IN_PARALLELISM": self.in_parallelism,
            "IN_NUM_PARALLELISM": self.in_num_parallelism,
            "W_PARALLELISM": self.w_parallelism,
            "W_NUM_PARALLELISM": self.w_num_parallelism,
            "IN_SIZE": self.in_size,
            "IN_DEPTH": self.in_depth,
        }

    def sw_compute(self):
        # get the matrix out result
        # from M[num_parallelism][depth],
        # and the element in M is m[parallelism][size]
        # to M_out[in1_num_parallelism][in2_num_parallelism]
        # the element in M_out is m_out[in1_parallelism][in2_parallelism]

        # collect all the input
        # breakpoint()
        data_out = self.att(self.x)
        output = self.data_pack(
            q2i(data_out, self.data_z_width, self.data_z_frac_width),
            self.in_num_parallelism,
            self.w_num_parallelism,
            self.in_parallelism,
            self.w_parallelism,
        )
        return output

    def att_data_generate(self):
        samples = self.samples
        # torch.manual_seed(0)
        in_y = self.in_num_parallelism * self.in_parallelism
        in_x = self.in_size * self.in_depth
        w_y = self.w_num_parallelism * self.w_parallelism
        self.x = torch.randn((samples, in_y, in_x))
        self.att = QuantizedAttention(
            dim=in_x,
            num_heads=1,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            config=self.w_config,
        )

        input_tensor = q2i(self.x, self.data_in_width, self.data_in_frac_width)
        wq = q2i(
            self.att.q.weight, self.weight_q_width, self.weight_q_frac_width
        ).repeat(samples, 1, 1)
        wkv = self.att.kv.weight.reshape(2, w_y, w_y)
        wk, wv = wkv[0], wkv[1]
        wk = q2i(wk, self.weight_k_width, self.weight_k_frac_width).repeat(
            samples, 1, 1
        )
        wv = q2i(wv, self.weight_v_width, self.weight_v_frac_width).repeat(
            samples, 1, 1
        )

        bq = q2i(self.att.q.bias, self.weight_q_width, self.weight_q_frac_width).repeat(
            samples, 1
        )
        bkv = self.att.kv.bias.reshape(2, w_y)
        bk, bv = bkv[0], bkv[1]
        bk = q2i(bk, self.bias_k_width, self.bias_k_frac_width).repeat(samples, 1)
        bv = q2i(bv, self.bias_v_width, self.bias_v_frac_width).repeat(samples, 1)

        in_num_parallelism = self.in_num_parallelism
        in_depth = self.in_depth
        in_parallelism = self.in_parallelism
        in_size = self.in_size
        w_parallelism = self.w_parallelism
        w_num_parallelism = self.w_num_parallelism

        data_in = self.data_pack(
            input_tensor, in_num_parallelism, in_depth, in_parallelism, in_size
        )
        wq_in = self.data_pack(wq, w_num_parallelism, in_depth, w_parallelism, in_size)
        wk_in = self.data_pack(wk, w_num_parallelism, in_depth, w_parallelism, in_size)
        wv_in = self.data_pack(wv, w_num_parallelism, in_depth, w_parallelism, in_size)

        bq_in = self.data_pack(bq, 1, w_num_parallelism, 1, w_parallelism)
        bk_in = self.data_pack(bk, 1, w_num_parallelism, 1, w_parallelism)
        bv_in = self.data_pack(bv, 1, w_num_parallelism, 1, w_parallelism)
        data_in.reverse()
        wq_in.reverse()
        wk_in.reverse()
        wv_in.reverse()
        bq_in.reverse()
        bk_in.reverse()
        bv_in.reverse()
        return (
            data_in,
            wq_in,
            wk_in,
            wv_in,
            bq_in,
            bk_in,
            bv_in,
        )

    def soft_max_data_generate(self, scale):
        # generate mem_init
        exp_table = generate_table_hardware(
            scale,
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
    samples = 30
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
    dut.bias_q_valid.value = 0
    dut.bias_k_valid.value = 0
    dut.bias_v_valid.value = 0
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
    for i in range(samples * 100):
        await FallingEdge(dut.clk)
        # breakpoint()
        dut.weight_q_valid.value = test_case.weight_q.pre_compute()
        dut.weight_k_valid.value = test_case.weight_k.pre_compute()
        dut.weight_v_valid.value = test_case.weight_v.pre_compute()
        dut.bias_q_valid.value = test_case.bias_q.pre_compute()
        dut.bias_k_valid.value = test_case.bias_k.pre_compute()
        dut.bias_v_valid.value = test_case.bias_v.pre_compute()
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

        dut.bias_q_valid.value, dut.bias_q.value = test_case.bias_q.compute(
            dut.bias_q_ready.value
        )
        dut.bias_k_valid.value, dut.bias_k.value = test_case.bias_k.compute(
            dut.bias_k_ready.value
        )
        dut.bias_v_valid.value, dut.bias_v.value = test_case.bias_v.compute(
            dut.bias_v_ready.value
        )

        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        # wave_check(dut)
        if (
            test_case.weight_q.is_empty()
            and test_case.weight_k.is_empty()
            and test_case.weight_v.is_empty()
            and test_case.bias_q.is_empty()
            and test_case.bias_k.is_empty()
            and test_case.bias_v.is_empty()
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
            dut.data_in_valid.value,
            [int(i) for i in dut.data_in.value],
            dut.data_out_valid.value,
            dut.data_out_valid.value,
            [int(i) for i in dut.data_out.value],
        )
    )


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
