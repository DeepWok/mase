#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging
import sys

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)

from mase_cocotb.runner import mase_runner
from utils import mxint_quantize

import torch
from math import ceil, log2
import random
import math

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MXIntDotProductTB(Testbench):
    def __init__(self, dut, num) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_0, dut.edata_in_0),
            dut.data_in_0_valid,
            dut.data_in_0_ready,
        )
        self.weight_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mweight, dut.eweight),
            dut.weight_valid,
            dut.weight_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
            signed=False,
        )

    def generate_inputs(self):
        inputs = []
        weights = []
        exp_outputs = []

        ebias_data = (2 ** (self.dut.DATA_IN_0_PRECISION_1.value - 1)) - 1
        ebias_weight = (2 ** (self.dut.WEIGHT_PRECISION_1.value - 1)) - 1
        ebias_out = (2 ** (self.dut.DATA_OUT_0_PRECISION_1.value - 1)) - 1

        w_man_w = self.dut.WEIGHT_PRECISION_0.value
        in_man_w = self.dut.DATA_IN_0_PRECISION_0.value
        out_man_w = self.dut.DATA_OUT_0_PRECISION_0.value

        fractional_point_adjustment = out_man_w - 2 - ((w_man_w - 2) + (in_man_w - 2))

        for _ in range(self.num):
            data = torch.randn(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = mxint_quantize(
                data,
                int(self.dut.DATA_IN_0_PRECISION_0),
                int(self.dut.DATA_IN_0_PRECISION_1),
            )
            w = torch.randn(int(self.dut.BLOCK_SIZE))
            (weight, mweight, eweight) = mxint_quantize(
                w,
                int(self.dut.WEIGHT_PRECISION_0),
                int(self.dut.WEIGHT_PRECISION_1),
            )

            # compute the mantissa
            mdp_out = 0
            for d, w in zip(mdata_in, mweight):
                mdp_out += math.floor(int(d) * int(w))
            # take the mod since the monitor comparison is unsigned
            mdp_out_unsigned = int(mdp_out) % (2**out_man_w)
            # adjust the exponent by the biases of the different widths
            edp_out = (edata_in - ebias_data) + (eweight - ebias_weight) + ebias_out
            # compute the result based on the exponent and mantissa found above
            out_manual = (mdp_out * 2 ** (-(w_man_w + in_man_w - 4))) * (
                2 ** (edp_out - ebias_out)
            )

            # compute the quantized output value in "full" precision
            out_q = data_in @ weight
            # check that the relative error doesn't exceed some amount
            assert (
                abs(out_q - out_manual) < 1e-6
            ), "Something went wrong when calculating the expected mantissa and exponents"

            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            weights.append((mweight.int().tolist(), eweight.int().tolist()))
            exp_outputs.append(
                (mdp_out_unsigned, int(edp_out + fractional_point_adjustment))
            )
        print(inputs)
        print(weights)
        print(exp_outputs)
        return inputs, weights, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, weights, exp_outputs = self.generate_inputs()

        # self.log.info(f"inputs: {inputs}\n{}")

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)
        self.weight_driver.load_driver(weights)
        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)
        # breakpoint()

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = MXIntDotProductTB(dut, num=50)
    await tb.run_test()


def get_config(seed):
    random.seed(seed)
    return {
        "DATA_IN_0_PRECISION_0": random.randint(2, 16),
        "DATA_IN_0_PRECISION_1": random.randint(2, 16),
        "WEIGHT_PRECISION_0": random.randint(2, 16),
        "WEIGHT_PRECISION_1": random.randint(2, 16),
        "BLOCK_SIZE": random.randint(2, 16),
    }


if __name__ == "__main__":
    torch.manual_seed(10)
    seed = os.getenv("COCOTB_SEED")

    if seed is not None:
        seed = int(seed)
        mase_runner(trace=True, module_param_list=[get_config(seed)])
    else:
        num_configs = int(os.getenv("NUM_CONFIGS", default=40))
        base_seed = random.randrange(sys.maxsize)
        mase_runner(
            trace=True,
            module_param_list=[get_config(base_seed + i) for i in range(num_configs)],
            jobs=10,
        )
        print(f"Test seeds: \n{[(i,base_seed+i) for i in range(num_configs)]}")
