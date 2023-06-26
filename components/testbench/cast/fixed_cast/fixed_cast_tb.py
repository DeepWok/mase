#!/usr/bin/env python3

# This script tests the fixed point castiplier
import random, os, logging

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


class VerificationCase:
    def __init__(self, samples=2):
        self.in_width = 6
        self.in_frac_width = 3
        self.out_width = 8
        self.out_frac_width = 4
        # self.in_width = 8
        # self.in_frac_width = 4
        # self.out_width = 6
        # self.out_frac_width = 3
        self.in_size = 1
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = [random.randint(0, 30) for _ in range(self.in_size)]
        return xs, self.sw_cast(xs)

    # TODO: This might not be efficient
    def sw_cast(self, inputs):
        outputs = []
        for i in range(0, len(inputs)):
            in_value = inputs[i]
            if self.in_frac_width > self.out_frac_width:
                in_value = in_value >> (self.in_frac_width - self.out_frac_width)
            else:
                in_value = in_value << (self.out_frac_width - self.in_frac_width)
            in_int_width = self.in_width - self.in_frac_width
            out_int_width = self.out_width - self.out_frac_width
            if in_int_width > out_int_width:
                if in_value >> (self.in_frac_width + out_int_width) > 0:
                    in_value = 1 << self.out_width - 1
                elif in_value >> (self.in_frac_width + out_int_width) < 0:
                    in_value = -(1 << self.out_width - 1)
                else:
                    in_value = int(in_value % (1 << self.out_width))
            outputs.append(in_value)
        # Hardware starts from MSB, which is opposite to software
        outputs.reverse()
        return outputs

    def get_dut_parameters(self):
        return {
            "IN_SIZE": self.in_size,
            "IN_WIDTH": self.in_width,
            "IN_FRAC_WIDTH": self.in_frac_width,
            "OUT_WIDTH": self.out_width,
            "OUT_FRAC_WIDTH": self.out_frac_width,
        }

    def get_dut_input(self, i):
        return self.inputs[i]

    def get_dut_output(self, i):
        return self.outputs[i]


def check_outputs(hw_out, sw_out):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
    for i in range(len(hw_out)):
        assert (
            hw_out[i] == sw_out[i]
        ), "Mismatched output value {}: {} expected = {}".format(
            i, int(hw_out[i]), sw_out[i]
        )
    return True


@cocotb.test()
async def test_fixed_adder_tree_layer(dut):
    """Test random number casting"""
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        dut.data_in.value = x
        await Timer(2, units="ns")
        assert check_outputs(
            [int(v) for v in dut.data_out.value], y
        ), "Output are incorrect on the {}th cycle: {} (expected {} => {})".format(
            i, [int(v) for v in dut.data_out.value], x, y
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../cast/fixed_cast.sv"]

    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_cast",
        build_args=extra_args,
    )
    runner.test(hdl_toplevel="fixed_cast", test_module="fixed_cast_tb")


if __name__ == "__main__":
    runner()
