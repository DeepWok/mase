#!/usr/bin/env python3

import os, pytest, random, sys

import torch
import logging
from functools import partial

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
)
from mase_cocotb.runner import mase_runner

torch.manual_seed(0)
# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from utils import MXIntRelu


class MXIntReluTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

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

        # Model
        self.model = MXIntRelu(
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "data_in_parallelism_dim_1": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_1"
                ),
                "data_in_parallelism_dim_0": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ),
            },
            bypass=True,
        )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        self.log.info(f"Mantissa Tensor: {mtensor}")
        self.log.info(f"Exponenr Tensor: {etensor}")
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return qtensor, tensor_inputs

    def generate_inputs(self):
        return torch.randn(
            (
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    async def run_test(self, us):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
        quantized, inputs = self.preprocess_tensor_for_mxint(
            tensor=inputs,
            config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_in_0_driver.load_driver(inputs)

        exp_out = self.model(quantized)
        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        _, outs = self.preprocess_tensor_for_mxint(
            tensor=exp_out,
            config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = MXIntReluTB(dut)
    await tb.run_test(us=100)


def get_relu_config(seed, kwargs={}):
    MAX_IN_FEATURES = 16
    MAX_BATCH_SIZE = 8
    random.seed(seed)

    BLOCK_SIZE = random.randint(2, 8)
    PARALLELISM = random.randint(1, 8)
    BATCH_SIZE = random.randint(1, MAX_BATCH_SIZE // PARALLELISM) * PARALLELISM
    IN_FEATURES = random.randint(2, MAX_IN_FEATURES // BLOCK_SIZE) * BLOCK_SIZE

    MAX_MANTISSA = 16
    MAX_EXPONENT = 6

    mantissa = random.randint(3, MAX_MANTISSA)
    exp = random.randint(3, min(mantissa, MAX_EXPONENT))


def get_relu_config(seed, kwargs={}):
    MAX_IN_FEATURES = 16
    MAX_BATCH_SIZE = 8
    random.seed(seed)

    BLOCK_SIZE = random.randint(2, 8)
    PARALLELISM = random.randint(1, 8)
    BATCH_SIZE = random.randint(1, MAX_BATCH_SIZE // PARALLELISM) * PARALLELISM
    IN_FEATURES = random.randint(2, MAX_IN_FEATURES // BLOCK_SIZE) * BLOCK_SIZE

    MAX_MANTISSA = 16
    MAX_EXPONENT = 6

    mantissa = random.randint(3, MAX_MANTISSA)
    exp = random.randint(3, min(mantissa, MAX_EXPONENT))

    config = {
        "DATA_IN_0_PRECISION_0": mantissa,
        "DATA_IN_0_PRECISION_1": exp,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": IN_FEATURES,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": BATCH_SIZE,
        "DATA_IN_0_PARALLELISM_DIM_0": BLOCK_SIZE,
        "DATA_IN_0_PARALLELISM_DIM_1": PARALLELISM,
    }

    config.update(kwargs)
    return config


@pytest.mark.dev
def test_relu():
    """
    Fully randomized parameter testing.
    """
    torch.manual_seed(10)
    seed = os.getenv("COCOTB_SEED")

    param_override = {}

    if seed is not None:
        seed = int(seed)
        mase_runner(
            trace=True,
            module_param_list=[get_relu_config(seed, param_override)],
        )
    else:
        num_configs = int(os.getenv("NUM_CONFIGS", default=1))
        base_seed = random.randrange(sys.maxsize)
        mase_runner(
            trace=True,
            module_param_list=[
                get_relu_config(base_seed + i, param_override)
                for i in range(num_configs)
            ],
            jobs=min(num_configs, os.cpu_count() // 2),
        )
        print(f"Test seeds: \n{[(i,base_seed+i) for i in range(num_configs)]}")


if __name__ == "__main__":
    test_relu()
