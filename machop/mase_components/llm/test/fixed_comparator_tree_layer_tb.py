#!/usr/bin/env python3

# This script tests the fixed point adder tree layer
import random, os

# Manually add mase_cocotb to system path
import sys, os
try:
    p = os.getenv("MASE_RTL")
    assert p != None
except:
    p = os.getenv("mase_rtl")
    assert p != None
p = os.path.join(p, '../')
sys.path.append(p)
##################################

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
from mase_cocotb.random_test import *

class VerificationCase:
    def __init__(self, samples=2):
        self.in_width = 32
        self.num = 9  # random.randint(2, 33)
        self.inputs, self.outputs = [], []  # length=samples
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = [random.randint(-30, 30) for _ in range(self.num)]
        return xs, self.sw_comparator_tree_layer(xs)

    def sw_comparator_tree_layer(self, inputs):
        outputs = []
        for i in range(0, int(len(inputs) / 2)):
            a = inputs[i]
            b = inputs[len(inputs) - i - 1]
            if (abs(a) > abs(b)):
                outputs.append(a)
            else:
                outputs.append(b)
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


@cocotb.test()
async def test_fixed_comparator_tree_layer(dut):
    """Test integer based adder tree layer"""
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i) 
        y = test_case.get_dut_output(i)  # SW outputs

        dut.data_in.value = x
        print(x)
        print(y)
        print(dut.data_out.value)
        await Timer(2, units="ns")
        check_results_signed(dut.data_out.value, y)
        # assert check_results_signed(
        #     [int(v) for v in dut.data_out.value], y
        # ), "Output are incorrect on the {}th cycle: {}".format(i, x)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
