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

from QAttention import QAttention

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 32
        self.data_in_frac_width = 4
        self.weight_width = 16
        self.weight_frac_width = 6

        self.in_parallelism = 3
        self.in_num_parallelism = 2

        self.in_size = 3
        self.in_depth = 2

        # notice:
        # require w is (dim * dim)
        # so wnp*wp = d*s
        # in test just set w_np = d, w_p = s
        self.w_parallelism = self.in_size
        self.w_num_parallelism = self.in_depth

        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.in_num_parallelism,
            debug=debug,
        )
        self.weight_q = RandomSource(
            name="weight_q",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            debug=debug,
        )
        self.weight_k = RandomSource(
            name="weight_k",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
            debug=debug,
        )
        self.weight_v = RandomSource(
            name="weight_v",
            samples=samples * self.in_depth * self.w_num_parallelism,
            num=self.w_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.w_num_parallelism,
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
            "WEIGHT_WIDTH": self.weight_width,
            "W_FRAC_WIDTH": self.weight_frac_width,
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
        d_temp = torch.tensor(self.data_in.data)
        wq_temp = torch.tensor(self.weight_q.data)
        wk_temp = torch.tensor(self.weight_k.data)
        wv_temp = torch.tensor(self.weight_v.data)

        d_tensor = self.data_arrange(
            d_temp,
            self.in_num_parallelism,
            self.in_depth,
            self.in_parallelism,
            self.in_size,
        )
        wq_tensor = self.data_arrange(
            wq_temp,
            self.w_num_parallelism,
            self.in_depth,
            self.w_parallelism,
            self.in_size,
        )
        wk_tensor = self.data_arrange(
            wk_temp,
            self.w_num_parallelism,
            self.in_depth,
            self.w_parallelism,
            self.in_size,
        )
        wv_tensor = self.data_arrange(
            wv_temp,
            self.w_num_parallelism,
            self.in_depth,
            self.w_parallelism,
            self.in_size,
        )
        logger.debug(
            "input data: \n\
        wq_tensor = \n{}\n wk_tensor = \n{}\n wv_tensor = \n{}\n\
        d_tensor = \n{}\n\
        ".format(
                wq_tensor, wk_tensor, wv_tensor, d_tensor
            )
        )
        # calculate the output
        # cut the output to smaller sets
        ref = []
        for i in range(self.samples):
            data = d_tensor[i]
            wq = wq_tensor[i]
            wk = wk_tensor[i]
            wv = wv_tensor[i]
            qatt = QAttention(
                data.shape[1],
                wq,
                wk,
                wv,
                self.data_in_width,
                self.data_in_frac_width,
                self.weight_width,
                self.weight_frac_width,
            )
            # calculate
            data = rearrange(data, "(b r) c ->b r c", b=1)
            out_temp = rearrange(qatt(data, data, data), "b r c->(b r) c ", b=1)

            # out pack
            output_tensor = self.output_pack(
                out_temp,
                self.in_num_parallelism,
                self.in_depth,
                self.in_parallelism,
                self.in_size,
            )
            # breakpoint()
            output = output_tensor.tolist()
            ref = ref + output
            # breakpoint()
        ref.reverse()
        return ref

    def data_arrange(self, in_temp, np, d, p, s):
        re_tensor = rearrange(
            in_temp,
            "(sa np d) (p s) -> (sa np) (d p) s",
            sa=self.samples,
            np=np,
            d=d,
            p=p,
            s=s,
        )

        ex_tensor = torch.zeros((self.samples * np), d * p, s)
        # print(d*p)
        for b in range(self.samples * np):
            for j in range(p):
                for i in range(d):
                    ex_tensor[b][j * d + i] = re_tensor[b][i * p + j]
        input_tensor = rearrange(
            ex_tensor,
            "(sa np) (p d) s -> sa (np p) (d s)",
            sa=self.samples,
            np=np,
            d=d,
            p=p,
            s=s,
        )
        return input_tensor

    def output_pack(self, out_temp, np, d, p, s):
        re_tensor = rearrange(
            out_temp, "(np p) (d s) -> np (p d) s", np=np, d=d, p=p, s=s
        )

        ex_tensor = torch.zeros(np, d * p, s)
        for b in range(np):
            for i in range(d):
                for j in range(p):
                    ex_tensor[b][i * p + j] = re_tensor[b][j * d + i]
        output_tensor = rearrange(
            ex_tensor, "np (d p) s -> (np d) (p s)", np=np, d=d, p=p, s=s
        )
        #   breakpoint()
        return output_tensor


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
    # logger.debug(
    #     "initial data:\n\
    # weight_q = {}\n\
    # weight_k = {}\n\
    # weight_v = {}\n\
    # data_in = {}\n".format(
    #         test_case.weight_q.data,
    #         test_case.weight_k.data,
    #         test_case.weight_v.data,
    #         test_case.data_in.data,
    #     )
    # )
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
    for i in range(samples * 50):
        await FallingEdge(dut.clk)
        # breakpoint()
        dut.weight_q_valid.value = test_case.weight_q.pre_compute()
        dut.weight_k_valid.value = test_case.weight_k.pre_compute()
        dut.weight_v_valid.value = test_case.weight_v.pre_compute()
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

    logger.debug(
        "wave of att:\n\
            {},{},data_in_q = {} \n\
            {},{},data_k = {} \n\
            {},{},data_v = {} \n\
            ".format(
            dut.att_inst.data_in_q_valid.value,
            dut.att_inst.data_in_q_valid.value,
            [int(i) for i in dut.att_inst.data_in_q.value],
            dut.att_inst.data_in_k_valid.value,
            dut.att_inst.data_in_k_valid.value,
            [int(i) for i in dut.att_inst.data_in_k.value],
            dut.att_inst.data_in_v_valid.value,
            dut.att_inst.data_in_v_valid.value,
            [int(i) for i in dut.att_inst.data_in_v.value],
        )
    )
    # logger.debug(
    #     "wave of matmul_z:\n\
    #         ib_data_in = {}\n\
    #         ib_data_in vr = {},{}\n\
    #         ib_weight = {}\n\
    #         ib_weight vr = {},{}\n\
    #         data_out = {}\n\
    #         data_out vr = {},{}\n\
    #         ".format(
    #         [int(i) for i in dut.inst_fmmc_z.ib_data_in.value],
    #         dut.inst_fmmc_z.ib_data_in_valid.value,
    #         dut.inst_fmmc_z.ib_data_in_ready.value,
    #         [int(i) for i in dut.inst_fmmc_z.ib_weight.value],
    #         dut.inst_fmmc_z.ib_weight_valid.value,
    #         dut.inst_fmmc_z.ib_weight_ready.value,
    #         [int(i) for i in dut.inst_fmmc_z.data_out.value],
    #         dut.inst_fmmc_z.data_out_valid.value,
    #         dut.inst_fmmc_z.data_out_ready.value,
    #     )
    # )

    # # data_out_no_cast = {}\n\
    # # [int(i) for i in dut.inst_fmmc_q.inst_fmmc.cast_data.value],
    # logger.debug(
    #     "wave of matmul_k:\n\
    #         ib_data_in = {}\n\
    #         ib_data_in vr = {},{}\n\
    #         ib_weight = {}\n\
    #         ib_weight vr = {},{}\n\
    #         data_out = {}\n\
    #         data_out vr = {},{}\n\
    #         ".format(
    #         [int(i) for i in dut.inst_fmmc_q.ib_data_in.value],
    #         dut.inst_fmmc_q.ib_data_in_valid.value,
    #         dut.inst_fmmc_q.ib_data_in_ready.value,
    #         [int(i) for i in dut.inst_fmmc_q.ib_weight.value],
    #         dut.inst_fmmc_q.ib_weight_valid.value,
    #         dut.inst_fmmc_q.ib_weight_ready.value,
    #         [int(i) for i in dut.inst_fmmc_q.data_out.value],
    #         dut.inst_fmmc_q.data_out_valid.value,
    #         dut.inst_fmmc_q.data_out_ready.value,
    #     )
    # )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/attention/fixed_self_att.sv",
        "../../../../components/attention/fixed_att.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/ram_block.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/linear/fixed_linear.sv",
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
        hdl_toplevel="fixed_self_att",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="fixed_self_att", test_module="fixed_self_att_tb")


if __name__ == "__main__":
    runner()
