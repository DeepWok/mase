#!/usr/bin/env python3

# This script tests the fixed point linear
import os
import logging
import sys
import pytest

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)

from utils import mxint_quantize
from mase_cocotb.runner import mase_runner

import torch
from math import ceil, log2
import random
from mase_cocotb.utils import bit_driver


logger = logging.getLogger("testbench")
logger.setLevel(logging.WARNING)

torch.manual_seed(10)


class MXIntAccumulatorTB(Testbench):
    def __init__(self, dut, num=1) -> None:
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
        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
            signed=False,
        )

    def generate_inputs(self):
        data_in = 20 * torch.randn(
            self.get_parameter("IN_DEPTH"), self.get_parameter("BLOCK_SIZE")
        )
        config_in = {
            "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
        }

        config_out = {
            "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
        }

        (data_in_q, tensor_out) = [], []
        for d in data_in:
            d_q, m_d, e_d = mxint_quantize(d, **config_in)
            data_in_q.append(d_q.tolist())
            tensor_out.append((m_d.int().tolist(), int(e_d)))

        # kinda jank but have to model the hardware exactly since accumulator doesn't normalize
        left_padding = ceil(log2(self.get_parameter("IN_DEPTH")))
        right_padding = config_out["width"] - config_in["width"] - left_padding
        # logical model of what the block is doing
        # accumulate and adjust the exponent
        mant_out, exp_max = tensor_out[0]
        mant_out = torch.tensor(mant_out, dtype=torch.float64)
        for mant, exp in tensor_out[1:]:
            mant = torch.tensor(mant, dtype=torch.float64)
            logger.debug(
                torch.remainder(mant_out * 2**right_padding, (2 ** config_out["width"]))
            )
            if exp > exp_max:
                mant_out = (mant_out * (2 ** (exp_max - exp))) + mant
                exp_max = exp
            else:
                mant_out += mant * (2 ** (exp - exp_max))
        logger.debug(
            torch.remainder(mant_out * 2**right_padding, (2 ** config_out["width"]))
        )
        m_data_out = torch.remainder(
            mant_out * 2 ** (right_padding), (2 ** config_out["width"])
        )
        e_bias_in = 2 ** (config_in["exponent_width"] - 1) - 1
        e_bias_out = 2 ** (config_out["exponent_width"] - 1) - 1
        e_data_out = exp_max - e_bias_in + e_bias_out + left_padding

        return tensor_out, [(m_data_out.to(torch.int64).tolist(), int(e_data_out))]

    async def run_test(self, samples, us):
        await self.reset()
        logger.info("Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for _ in range(samples):
            logger.info("generating inputs")
            inputs, exp_outputs = self.generate_inputs()
            logger.info(f"{inputs} {exp_outputs}")

            # Load the inputs driver
            self.data_in_0_driver.load_driver(inputs)
            # Load the output monitor
            self.data_out_0_monitor.load_monitor(exp_outputs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


# @cocotb.test()
# async def test(dut):
#     tb = MXIntAccumulatorTB(dut, 1)
#     await tb.run_test(samples=20, us=5)

# @cocotb.test()
# async def single_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1, us=100)


# @cocotb.test()
# async def repeated_mult(dut):
#     tb = MXIntMatmulTB(dut)
#     tb.output_monitor.ready.value = 1
#     await tb.run_test(batches=1000, us=2000)


@cocotb.test()
async def repeated_mult_valid_backpressure(dut):
    tb = MXIntAccumulatorTB(dut, 10)
    tb.data_in_0_driver.set_valid_prob(0.7)
    cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, 0.6))
    num_samples = 100
    await tb.run_test(
        samples=num_samples, us=0.05 * num_samples * tb.get_parameter("IN_DEPTH")
    )


def get_config(seed):
    random.seed(seed)
    return {
        "DATA_IN_0_PRECISION_0": random.randint(2, 16),
        "DATA_IN_0_PRECISION_1": random.randint(2, 4),
        "BLOCK_SIZE": random.randint(2, 16),
        "IN_DEPTH": random.randint(1, 100),
    }


@pytest.mark.dev
def run_random_tests():
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
            jobs=min(num_configs, 10),
        )
        print(f"Test seeds: \n{[(i, base_seed + i) for i in range(num_configs)]}")


if __name__ == "__main__":
    run_random_tests()
