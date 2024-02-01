#!/usr/bin/env python3

# This script tests the register slice
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
import torch
from pvt_quant import fixed_affine
from z_qlayers import quantize_to_int as q2i

debug = True
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.data_width = 8
        self.data_frac_width = 3
        self.b_width = 8
        self.b_frac_width = 5
        self.data_out_width = 8
        self.data_out_frac_width = 4
        self.w_config = {
            "affine": {
                "mul": {
                    "name": "integer",
                    "data_in_width": self.data_width,
                    "data_in_frac_width": self.data_frac_width,
                },
                "add": {
                    "name": "integer",
                    "data_in_width": self.b_width,
                    "data_in_frac_width": self.b_frac_width,
                },
            }
        }
        self.in_size = 4
        self.data_generate()
        self.inputs = RandomSource(
            samples=samples,
            max_stalls=2 * samples,
            num=self.in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=self.data_in,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def data_generate(self):
        torch.manual_seed(0)
        samples = self.samples
        self.fixed_aff = fixed_affine(self.w_config["affine"])
        self.x = 3 * torch.randn(self.samples, self.in_size)
        w = self.fixed_aff.weight
        b = self.fixed_aff.bias
        self.weight_in = (
            q2i(
                w,
                self.w_config["affine"]["mul"]["data_in_width"],
                self.w_config["affine"]["mul"]["data_in_frac_width"],
            )
            .repeat(self.samples, self.in_size)
            .tolist()
        )

        self.bias_in = (
            q2i(
                b,
                self.w_config["affine"]["add"]["data_in_width"],
                self.w_config["affine"]["add"]["data_in_frac_width"],
            )
            .repeat(self.samples, self.in_size)
            .tolist()
        )
        self.data_in = q2i(
            self.x,
            self.w_config["affine"]["mul"]["data_in_width"],
            self.w_config["affine"]["mul"]["data_in_frac_width"],
        ).tolist()
        self.data_in.reverse()
        self.weight_in.reverse()
        self.bias_in.reverse()
        self.weight = RandomSource(
            samples=samples,
            max_stalls=2 * samples,
            num=self.in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=self.weight_in,
        )
        self.bias = RandomSource(
            samples=samples,
            max_stalls=2 * samples,
            num=self.in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=self.bias_in,
        )

    def sw_compute(self):
        data_out = self.fixed_aff(self.x)
        output = q2i(data_out, self.data_out_width, self.data_out_frac_width).tolist()
        return output

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["affine"]["mul"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["affine"]["mul"]["data_in_frac_width"],
            "BIAS_WIDTH": self.w_config["affine"]["add"]["data_in_width"],
            "BIAS_FRAC_WIDTH": self.w_config["affine"]["add"]["data_in_frac_width"],
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN_SIZE": self.in_size,
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
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    done = False
    while not done:
        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")

        ## Pre_compute
        dut.data_in_valid.value = test_case.inputs.pre_compute()
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.bias_valid.value = test_case.bias.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")

        ## Compute
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        # wave_check(dut)
        logger.debug("\n")
        # breakpoint()
        done = (
            test_case.outputs.is_full()
            and test_case.inputs.is_empty()
            and test_case.weight.is_empty()
            and test_case.bias.is_empty()
        )

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave of in_out:\n\
            {},{},data_in = {} \n\
            {},{},weight_in = {} \n\
            {},{},bias = {} \n\
            {},{},data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.weight_valid.value,
            dut.weight_ready.value,
            [int(i) for i in dut.weight.value],
            dut.bias_valid.value,
            dut.bias_ready.value,
            [int(i) for i in dut.bias.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )

    logger.debug(
        "wave of sa_out:\n\
            {},{},prod = {} \n\
            {},{},add  = {} \n\
            ".format(
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.prod.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.round_prod.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.round_in.value],
        )
    )
    breakpoint()


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/affine_layernorm.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/fixed_arithmetic/fixed_vector_mult.sv",
        "../../../../components/fixed_arithmetic/fixed_mult.sv",
        "../../../../components/common/fifo.sv",
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
        hdl_toplevel="affine_layernorm",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="affine_layernorm", test_module="affine_layernorm_tb")


if __name__ == "__main__":
    runner()
