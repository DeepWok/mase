#!/usr/bin/env python3

# This script tests the fixed point adder tree layer
import random, os

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner


class VerificationCase:
    def __init__(self, samples=2):
        self.in_width = 32
        self.num = 17  # random.randint(2, 33)
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = [random.randint(0, 30) for _ in range(self.num)]
        return xs, self.sw_adder_tree(xs)

    def sw_adder_tree(self, inputs):
        outputs = []
        for i in range(0, int(len(inputs) / 2)):
            outputs.append(inputs[i] + inputs[len(inputs) - i - 1])
        if len(inputs) % 2:
            outputs.append(inputs[int(len(inputs) / 2)])
        # Hardware starts from MSB, which is opposite to software
        outputs.reverse()
        return outputs

    def get_dut_parameters(self):
        return {
            "IN_SIZE": self.num,
            "IN_WIDTH": self.in_width,
        }

    def get_dut_input(self, i):
        return self.inputs[i]

    def get_dut_output(self, i):
        return self.outputs[i]


def check_outputs(hw_out, sw_out):
    if len(hw_out) != len(sw_out):
        print(
            "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
        )
        return False
    for i in range(len(hw_out)):
        if hw_out[i] != sw_out[i]:
            print(
                "Mismatched output value {}: {} expected = {}".format(
                    i, int(hw_out[i]), sw_out[i]
                )
            )
            return False
    return True


@cocotb.test()
async def test_fixed_adder_tree_layer(dut):
    """Test integer based adder tree layer"""
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in.value = x
        await Timer(2, units="ns")
        assert check_outputs(
            [int(v) for v in dut.data_out.value], y
        ), "Output are incorrect on the {}th cycle: {}".format(i, x)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = ["../../../../hardware/fixed_arith/fixed_adder_tree_layer.sv"]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(
        verilog_sources=verilog_sources,
        toplevel="fixed_adder_tree_layer",
        extra_args=extra_args,
    )

    runner.test(
        toplevel="fixed_adder_tree_layer", py_module="fixed_adder_tree_layer_tb"
    )


if __name__ == "__main__":
    runner()
