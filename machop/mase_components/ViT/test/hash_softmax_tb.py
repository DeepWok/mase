#!/usr/bin/env python3

# This script tests the register slice
import random, os, math, logging, sys, torch

sys.path.append("/workspace/components/testbench/ViT")
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from z_qlayers import quantize_to_int as q2i
from einops import rearrange
from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner
from ha_softmax import (
    generate_table_div_hardware,
    generate_table_hardware,
    QHashSoftmax,
)

debug = False
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.w_config = {
            "softmax": {
                "exp_width": 8,
                "exp_frac_width": 4,
                "div_width": 10,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_out_width": 8,
                "data_out_frac_width": 6,
            },
        }
        self.d_config = {
            "softmax": {
                "in_size": 1,
                "out_size": 1,
                "in_depth": 4,
            },
        }
        in_size = self.d_config["softmax"]["in_size"]
        out_size = self.d_config["softmax"]["out_size"]
        in_depth = self.d_config["softmax"]["in_depth"]
        self.samples = samples
        self.data_generate()
        self.inputs = RandomSource(
            name="data_in",
            samples=samples * in_depth,
            max_stalls=0,
            num=in_size,
            data_specify=self.d_in,
            debug=debug,
        )
        self.outputs = RandomSink(
            samples=samples * in_size * in_depth // out_size, max_stalls=0, debug=debug
        )
        self.ref = self.sw_compute()

    def data_generate(self):
        B = self.samples
        C = self.d_config["softmax"]["in_size"] * self.d_config["softmax"]["in_depth"]

        torch.manual_seed(0)
        self.x = 5 * torch.randn(B, C)
        x_in = q2i(
            self.x,
            self.w_config["softmax"]["data_in_width"],
            self.w_config["softmax"]["data_in_frac_width"],
        )
        exp_table = generate_table_hardware(
            1,
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
        self.qhsoftmax = QHashSoftmax(self.w_config["softmax"])
        # data_pack
        self.d_in = self.linear_data_pack(
            x_in, 1, C, 1, self.d_config["softmax"]["in_size"]
        )
        self.d_in.reverse()

    def sw_compute(self):
        data_out = self.qhsoftmax(self.x, 1)
        output = q2i(
            data_out,
            self.w_config["softmax"]["data_out_width"],
            self.w_config["softmax"]["data_out_frac_width"],
        )

        C = self.d_config["softmax"]["in_size"] * self.d_config["softmax"]["in_depth"]
        output = self.linear_data_pack(
            output, 1, C, 1, self.d_config["softmax"]["out_size"]
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

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["softmax"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["softmax"]["data_in_frac_width"],
            "EXP_WIDTH": self.w_config["softmax"]["exp_width"],
            "EXP_FRAC_WIDTH": self.w_config["softmax"]["exp_frac_width"],
            "DIV_WIDTH": self.w_config["softmax"]["div_width"],
            "OUT_WIDTH": self.w_config["softmax"]["data_out_width"],
            "OUT_FRAC_WIDTH": self.w_config["softmax"]["data_out_frac_width"],
            "IN_SIZE": self.d_config["softmax"]["in_size"],
            "OUT_SIZE": self.d_config["softmax"]["out_size"],
            "IN_DEPTH": self.d_config["softmax"]["in_depth"],
        }


def in_out_wave(dut, name):
    logger.debug(
        "{}  State: (in_valid,in_ready,out_valid,out_ready) = ({},{},{},{})".format(
            name,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_register_slice(dut):
    """Test register slice"""
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
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    await FallingEdge(dut.clk)

    await FallingEdge(dut.clk)

    done = False
    while not done:
        await FallingEdge(dut.clk)

        ## Pre_compute
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        ## Compute
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        logger.debug(
            "\n\
                     {}{}in = {}\n\
                     {}{}sum={}\n\
                     {}{}acc={}\n\
                     circular_count={}\n\
                     {}{}ib_acc={}\n\
                     {}{}out= {}\n\
                     ".format(
                dut.data_in_valid.value,
                dut.data_in_ready.value,
                [int(i) for i in dut.data_in.value],
                dut.sum_valid.value,
                dut.sum_ready.value,
                int(dut.sum.value),
                dut.acc_valid.value,
                dut.acc_ready.value,
                [int(i) for i in dut.acc_duplicate.value],
                int(dut.acc_circular.circular_count.value),
                dut.ib_acc_valid.value,
                dut.ib_acc_ready.value,
                [int(i) for i in dut.ib_acc.value],
                dut.data_out_valid.value,
                dut.data_out_ready.value,
                [int(i) for i in dut.data_out.value],
            )
        )
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/hash_softmax.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arithmetic/fixed_accumulator.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/split2.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/cast/fixed_rounding.sv",
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
        hdl_toplevel="hash_softmax",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="hash_softmax", test_module="hash_softmax_tb")


if __name__ == "__main__":
    runner()
