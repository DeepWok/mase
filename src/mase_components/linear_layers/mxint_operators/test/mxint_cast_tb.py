#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging, pytest

import random
import sys
from mase_cocotb.utils import bit_driver

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

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MXINTVectorMultTB(Testbench):
    def __init__(self, dut, num) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.num = num
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

        self.data_in_0_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in, dut.edata_in),
            dut.data_in_valid,
            dut.data_in_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out, dut.edata_out),
            dut.data_out_valid,
            dut.data_out_ready,
            check=True,
            off_by_one=True,
        )

        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_outputs(self, mdata_in, edata_in):
        block_size = int(self.get_parameter("BLOCK_SIZE"))
        mantissa_width_in = int(self.get_parameter("IN_MAN_WIDTH"))
        exponent_width_in = int(self.get_parameter("IN_EXP_WIDTH"))
        mantissa_width_out = int(self.get_parameter("OUT_MAN_WIDTH"))
        exponent_width_out = int(self.get_parameter("OUT_EXP_WIDTH"))

        data_in = torch.tensor(mdata_in, dtype=torch.float) * (
            2 ** (2 - mantissa_width_in + edata_in - 2 ** (exponent_width_in - 1) + 1)
        )
        data_out, mdata_out, edata_out = mxint_quantize(
            data_in, mantissa_width_out, exponent_width_out
        )

        mdata_out = mdata_out.int().tolist()
        edata_out = edata_out.int().item()

        bit_mask = 2 ** (exponent_width_out - 1) - 1
        edata_out = (edata_out & bit_mask) - (edata_out & ~bit_mask)

        return mdata_out, edata_out

    def generate_inputs(self):

        block_size = int(self.get_parameter("BLOCK_SIZE"))
        mantissa_width = int(self.get_parameter("IN_MAN_WIDTH"))
        exponent_width = int(self.get_parameter("IN_EXP_WIDTH"))

        inputs = []
        exp_outputs = []

        for _ in range(self.num):
            mdata_in = [
                random.randint(
                    -(2 ** (mantissa_width - 1)) + 1, 2 ** (mantissa_width - 1) - 1
                )
                for _ in range(block_size)
            ]
            edata_in = random.randint(0, 2**exponent_width - 1)

            inputs.append((mdata_in, edata_in))

            mdata_out, edata_out = self.generate_outputs(mdata_in, edata_in)

            exp_outputs.append((mdata_out, edata_out))

        return inputs, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        logger.info(f"generating inputs")
        inputs, exp_outputs = self.generate_inputs()

        self.data_in_0_driver.load_driver(inputs)

        self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(5, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


def get_mxint_cast_config_random(seed, kwargs={}):
    random.seed(seed)

    BLOCK_SIZE = random.randint(1, 16)

    MAX_MANTISSA = 32
    MAX_EXPONENT = 6

    in_man = random.randint(3, MAX_MANTISSA)
    in_exp = random.randint(2, MAX_EXPONENT)

    out_man = random.randint(3, MAX_MANTISSA)
    out_exp = random.randint(2, MAX_EXPONENT)

    config = {
        "IN_MAN_WIDTH": in_man,
        "IN_EXP_WIDTH": in_exp,
        "OUT_MAN_WIDTH": out_man,
        "OUT_EXP_WIDTH": out_exp,
        "BLOCK_SIZE": BLOCK_SIZE,
    }

    config.update(kwargs)
    return config


@pytest.mark.dev
def test_mxint_cast_random():
    """
    Fully randomized parameter testing.
    """
    torch.manual_seed(10)
    seed = os.getenv("COCOTB_SEED")

    # use this to fix a particular parameter value
    param_override = {
        # "IN_MAN_WIDTH": 32,
        # "IN_EXP_WIDTH": 6,
        # "OUT_MAN_WIDTH": 6,
        # "OUT_EXP_WIDTH": 3,
        # "BLOCK_SIZE": 2,
    }

    if seed is not None:
        seed = int(seed)
        mase_runner(
            trace=True,
            module_param_list=[get_mxint_cast_config_random(seed, param_override)],
        )
    else:
        num_configs = int(os.getenv("NUM_CONFIGS", default=1))
        base_seed = random.randrange(sys.maxsize)
        mase_runner(
            trace=True,
            module_param_list=[
                get_mxint_cast_config_random(base_seed + i, param_override)
                for i in range(num_configs)
            ],
            jobs=min(num_configs, os.cpu_count() // 2),
        )
        print(f"Test seeds: \n{[(i,base_seed+i) for i in range(num_configs)]}")


@cocotb.test()
async def test(dut):
    tb = MXINTVectorMultTB(dut, num=100)
    await tb.run_test()


if __name__ == "__main__":
    test_mxint_cast_random()
