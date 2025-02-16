#!/usr/bin/env python3

# This script tests the fixed point castiplier
import logging
import cocotb
import pytest
from cocotb.triggers import Timer

import torch
from chop.nn.quantizers import integer_floor_quantizer
from mase_cocotb.z_qlayers import _integer_quantize
from mase_cocotb.z_qlayers import quantize_to_int as q2i
from mase_cocotb.runner import mase_runner

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


class VerificationCase:
    def __init__(self, samples=2):
        self.in_width = 17
        self.in_frac_width = 8
        self.out_width = 16
        self.out_frac_width = 8
        self.in_size = 10
        self.inputs, self.outputs = [], []
        torch.manual_seed(0)
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        data = torch.randn(self.in_size)
        data = integer_floor_quantizer(data, self.in_width, self.in_frac_width)
        print("data = ", data)
        h = q2i(data, self.in_width, self.in_frac_width)
        return [int(i) for i in h], [int(i) for i in self.sw_cast(data)]

    def sw_cast(self, inputs):
        outputs = (
            integer_floor_quantizer(inputs, self.out_width, self.out_frac_width)
            * 2**self.out_frac_width
        )
        # breakpoint()
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
        if hw_out[i] != sw_out[i]:
            return False
    return True


@cocotb.test()
async def cocotb_test_fixed_rounding(dut):
    """Test random number casting"""
    test_case = VerificationCase(samples=200)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        dut.data_in.value = x
        await Timer(2, units="ns")
        assert check_outputs(
            [v.signed_integer for v in dut.data_out.value], y
        ), "Output are incorrect on the {}th cycle: {} (expected {} => {})".format(
            i, [v.signed_integer for v in dut.data_out.value], x, y
        )


@pytest.mark.dev
def test_fixed_rounding():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_fixed_rounding()
