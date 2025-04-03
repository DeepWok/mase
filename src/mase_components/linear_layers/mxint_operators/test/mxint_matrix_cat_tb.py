#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging, pytest

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)

from mase_cocotb.runner import mase_runner
from utils import mxint_quantize, block_mxint_quant, pack_tensor_to_mx_listed_chunk
import sys

from chop.nn.quantizers.mxint import _mxint_quantize

import torch
from math import ceil, log2
import random

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(10)


class MXIntMatrixCat(Testbench):
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

        self.data_in_1_driver = MultiSignalStreamDriver(
            dut.clk,
            (dut.mdata_in_1, dut.edata_in_1),
            dut.data_in_1_valid,
            dut.data_in_1_ready,
        )

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
        )
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_in_1_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):

        din1 = []
        din2 = []
        exp_outputs = []

        for _ in range(self.num):

            d0 = torch.rand(
                int(self.dut.DATA_IN_0_TENSOR_SIZE_DIM_0),
                int(self.dut.DATA_IN_0_TENSOR_SIZE_DIM_1),
            )

            d1 = torch.rand(
                int(self.dut.DATA_IN_1_TENSOR_SIZE_DIM_0),
                int(self.dut.DATA_IN_1_TENSOR_SIZE_DIM_1),
            )

            (data_in_0, mdata_in_0, edata_in_0) = block_mxint_quant(
                d0,
                {
                    "width": int(self.dut.DATA_IN_0_PRECISION_0),
                    "exponent_width": int(self.dut.DATA_IN_0_PRECISION_1),
                },
                [
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_0_PARALLELISM_DIM_1),
                ],
            )

            (data_in_1, mdata_in_1, edata_in_1) = block_mxint_quant(
                d1,
                {
                    "width": int(self.dut.DATA_IN_1_PRECISION_0),
                    "exponent_width": int(self.dut.DATA_IN_1_PRECISION_1),
                },
                [
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_0),
                    int(self.dut.DATA_IN_1_PARALLELISM_DIM_1),
                ],
            )

            din1.append(
                pack_tensor_to_mx_listed_chunk(
                    mdata_in_0,
                    edata_in_0,
                    [
                        int(self.dut.DATA_IN_0_PARALLELISM_DIM_0),
                        int(self.dut.DATA_IN_0_PARALLELISM_DIM_1),
                    ],
                )
            )

            din2.append(
                pack_tensor_to_mx_listed_chunk(
                    mdata_in_1,
                    edata_in_1,
                    [
                        int(self.dut.DATA_IN_1_PARALLELISM_DIM_0),
                        int(self.dut.DATA_IN_1_PARALLELISM_DIM_1),
                    ],
                )
            )

            mdp = torch.cat([d0, d1], dim=-1)

            (data_out_0, mdata_out_0, edata_out_0) = block_mxint_quant(
                mdp,
                {
                    "width": int(self.dut.DATA_OUT_0_PRECISION_0),
                    "exponent_width": int(self.dut.DATA_OUT_0_PRECISION_1),
                },
                [
                    int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0),
                    int(self.dut.DATA_OUT_0_PARALLELISM_DIM_1),
                ],
            )

            exp_outputs.append(
                pack_tensor_to_mx_listed_chunk(
                    mdata_out_0,
                    edata_out_0,
                    [
                        int(self.dut.DATA_OUT_0_PARALLELISM_DIM_0),
                        int(self.dut.DATA_OUT_0_PARALLELISM_DIM_1),
                    ],
                )
            )

        print(f"Din 1: \n {din1}")
        print(f"Din 2: \n {din2}")
        print(f"Dout: \n {exp_outputs}")

        return din1, din2, exp_outputs

    async def run_test(self):
        await self.reset()

        for i in range(0, self.num):

            logger.info(f"Reset finished")

            self.data_out_0_monitor.ready.value = 1

            logger.info(f"generating inputs")
            inputs, weights, exp_outputs = self.generate_inputs()

            # Load the inputs driver
            self.data_in_0_driver.load_driver(inputs[i])
            self.data_in_1_driver.load_driver(weights[i])
            # # Load the output monitor
            self.data_out_0_monitor.load_monitor(exp_outputs[i])

        await Timer(100, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = MXIntMatrixCat(dut, num=1)
    await tb.run_test()


def get_config(seed: int):
    random.seed(seed)

    MAX_MANTISSA = 16
    MAX_EXPONENT = 6

    cat_dim = random.randint(4, 16)

    dim_din_0 = random.randint(4, 16)
    dim_din_1 = random.randint(1, 10) * dim_din_0

    factors = [i for i in range(1, cat_dim + 1) if cat_dim % i == 0]
    parallelism_c = random.choice(factors)

    factors = [i for i in range(1, dim_din_0 + 1) if dim_din_0 % i == 0]
    parallelism_0 = random.choice(factors)

    prec_0_0 = random.randint(4, 16)
    prec_0_1 = random.randint(4, MAX_EXPONENT)

    prec_1_0 = random.randint(4, 16)
    prec_1_1 = random.randint(4, MAX_EXPONENT)

    prec_out_0 = random.randint(3, min([prec_0_0, prec_0_1, prec_1_0, prec_1_1]))
    prec_out_1 = random.randint(3, min([prec_0_0, prec_0_1, prec_1_0, prec_1_1]))

    config = {
        "DATA_IN_0_PRECISION_0": prec_0_0,
        "DATA_IN_0_PRECISION_1": prec_0_1,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": cat_dim,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": dim_din_0,
        "DATA_IN_0_PARALLELISM_DIM_0": parallelism_c,
        "DATA_IN_0_PARALLELISM_DIM_1": parallelism_0,
        "DATA_IN_1_PRECISION_0": prec_1_0,
        "DATA_IN_1_PRECISION_1": prec_1_1,
        "DATA_IN_1_TENSOR_SIZE_DIM_0": cat_dim,
        "DATA_IN_1_TENSOR_SIZE_DIM_1": dim_din_1,
        "DATA_IN_1_PARALLELISM_DIM_0": parallelism_c,
        "DATA_IN_1_PARALLELISM_DIM_1": parallelism_0,
        "DATA_OUT_0_PRECISION_0": prec_out_0,
        "DATA_OUT_0_PRECISION_1": prec_out_1,
    }

    return config


@pytest.mark.dev
def test_random_cat():
    torch.manual_seed(1)
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
    test_random_cat()
