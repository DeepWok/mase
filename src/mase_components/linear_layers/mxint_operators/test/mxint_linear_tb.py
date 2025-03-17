#!/usr/bin/env python3

import random
import sys
from mase_cocotb.utils import bit_driver
import os, pytest

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
from utils import MXIntLinear


class LinearTB(Testbench):
    def __init__(
        self, dut, data_in_p=1.0, weight_p=1.0, bias_p=1.0, data_out_p=1.0
    ) -> None:
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
        self.data_in_0_driver.set_valid_prob(data_in_p)

        self.weight_driver = MultiSignalStreamDriver(
            dut.clk, (dut.mweight, dut.eweight), dut.weight_valid, dut.weight_ready
        )
        self.weight_driver.set_valid_prob(weight_p)

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_driver = MultiSignalStreamDriver(
                dut.clk, (dut.mbias, dut.ebias), dut.bias_valid, dut.bias_ready
            )
            self.bias_driver.log.setLevel(logging.DEBUG)
            self.bias_driver.set_valid_prob(bias_p)

        self.data_out_0_monitor = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=True,
            signed=False,
            off_by_one=True,
        )
        cocotb.start_soon(bit_driver(dut.data_out_0_ready, dut.clk, data_out_p))

        # Model
        self.model = MXIntLinear(
            in_features=self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            out_features=self.get_parameter("DATA_OUT_0_TENSOR_SIZE_DIM_0"),
            bias=True if self.get_parameter("HAS_BIAS") == 1 else False,
            config={
                "data_in_width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "data_in_exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                "data_in_parallelism_dim_1": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_1"
                ),
                "data_in_parallelism_dim_0": self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ),
                "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
                "weight_exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
                "weight_parallelism_dim_1": self.get_parameter(
                    "WEIGHT_PARALLELISM_DIM_1"
                ),
                "weight_parallelism_dim_0": self.get_parameter(
                    "WEIGHT_PARALLELISM_DIM_0"
                ),
                "bias_width": self.get_parameter("BIAS_PRECISION_0"),
                "bias_exponent_width": self.get_parameter("BIAS_PRECISION_1"),
                "bias_parallelism_dim_1": self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                "bias_parallelism_dim_0": self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                "data_out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "data_out_exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                "data_out_parallelism_dim_1": self.get_parameter(
                    "DATA_OUT_0_PARALLELISM_DIM_1"
                ),
                "data_out_parallelism_dim_0": self.get_parameter(
                    "DATA_OUT_0_PARALLELISM_DIM_0"
                ),
            },
        )

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from utils import block_mxint_quant
        from utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = block_mxint_quant(tensor, config, parallelism)
        # take the mod to get the unsigned representation of the number
        mtensor = mtensor.remainder(2 ** config["width"])
        self.log.info(f"Mantissa Tensor: {mtensor}")
        self.log.info(f"Exponent Tensor: {etensor}")

        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

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
        exp_out = self.model(inputs)

        # * Load the inputs driver
        self.log.info(f"Processing inputs: {inputs}")
        inputs_quant = self.preprocess_tensor_for_mxint(
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
        self.data_in_0_driver.load_driver(inputs_quant)
        self.log.info(f"processed inputs\n{inputs_quant}")
        # * Load the weights driver
        weights = self.model.weight

        self.log.info(f"Processing weights:\n{weights}")
        weights = self.preprocess_tensor_for_mxint(
            tensor=weights,
            config={
                "width": self.get_parameter("WEIGHT_PRECISION_0"),
                "exponent_width": self.get_parameter("WEIGHT_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("WEIGHT_PARALLELISM_DIM_1"),
                self.get_parameter("WEIGHT_PARALLELISM_DIM_0"),
            ],
        )
        self.log.info(f"Processed weights:\n{weights}")
        self.weight_driver.load_driver(weights)

        # * Load the bias driver
        if self.get_parameter("HAS_BIAS") == 1:
            bias = self.model.bias
            self.log.info(f"Processing bias: {bias}")
            bias = self.preprocess_tensor_for_mxint(
                tensor=bias,
                config={
                    "width": self.get_parameter("BIAS_PRECISION_0"),
                    "exponent_width": self.get_parameter("BIAS_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                    self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                ],
            )
            self.bias_driver.load_driver(bias)

        # * Load the output monitor
        self.log.info(f"Processing outputs: {exp_out}")
        outs = self.preprocess_tensor_for_mxint(
            tensor=exp_out,
            config={
                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(us, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    probs = [torch.rand(1).item() for _ in range(4)]
    # probs = [1 for _ in range(4)]
    tb = LinearTB(
        dut,
        data_in_p=probs[0],
        weight_p=probs[1],
        bias_p=probs[2],
        data_out_p=probs[3],
    )
    await tb.run_test(us=200)


def get_mxint_linear_config_random(seed, kwargs={}):
    MAX_IN_FEATURES = 16
    MAX_OUT_FEATURES = 16
    MAX_BATCH_SIZE = 8
    random.seed(seed)

    BLOCK_SIZE = random.randint(2, 8)
    PARALLELISM = random.randint(1, 8)
    BATCH_SIZE = random.randint(1, MAX_BATCH_SIZE // PARALLELISM) * PARALLELISM
    IN_FEATURES = random.randint(2, MAX_IN_FEATURES // BLOCK_SIZE) * BLOCK_SIZE
    OUT_FEATURES = random.randint(2, MAX_OUT_FEATURES // BLOCK_SIZE) * BLOCK_SIZE

    MAX_MANTISSA = 16
    MAX_EXPONENT = 6

    mantissas = [random.randint(3, MAX_MANTISSA)] * 4
    exps = [random.randint(3, min(mantissas[0], MAX_EXPONENT))] * 4

    config = {
        "HAS_BIAS": random.randint(0, 1),
        "DATA_IN_0_TENSOR_SIZE_DIM_0": IN_FEATURES,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": BATCH_SIZE,
        "DATA_IN_0_PARALLELISM_DIM_0": BLOCK_SIZE,
        "DATA_IN_0_PARALLELISM_DIM_1": PARALLELISM,
        "WEIGHT_TENSOR_SIZE_DIM_0": IN_FEATURES,
        "WEIGHT_TENSOR_SIZE_DIM_1": OUT_FEATURES,
        "WEIGHT_PARALLELISM_DIM_0": BLOCK_SIZE,
        "WEIGHT_PARALLELISM_DIM_1": BLOCK_SIZE,
        "DATA_IN_0_PRECISION_0": mantissas[0],
        "DATA_IN_0_PRECISION_1": exps[0],
        "WEIGHT_PRECISION_0": mantissas[1],
        "WEIGHT_PRECISION_1": exps[1],
        "BIAS_PRECISION_0": mantissas[2],
        "BIAS_PRECISION_1": exps[2],
        "DATA_OUT_0_PRECISION_0": mantissas[3],
        "DATA_OUT_0_PRECISION_1": exps[3],
    }
    config.update(kwargs)
    return config


def get_mxint_linear_config(kwargs={}):
    config = {
        "HAS_BIAS": 1,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 2,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 2,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": 2,
        "WEIGHT_TENSOR_SIZE_DIM_1": 2,
        "WEIGHT_PARALLELISM_DIM_0": 2,
        "WEIGHT_PARALLELISM_DIM_1": 1,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 4,
        "WEIGHT_PRECISION_0": 8,
        "WEIGHT_PRECISION_1": 4,
        "BIAS_PRECISION_0": 8,
        "BIAS_PRECISION_1": 4,
        "DATA_OUT_0_PRECISION_0": 10,
        "DATA_OUT_0_PRECISION_1": 4,
    }
    config.update(kwargs)
    return config


@pytest.mark.dev
def test_mxint_linear_full_random():
    """
    Fully randomized parameter testing.
    """
    torch.manual_seed(10)
    seed = os.getenv("COCOTB_SEED")

    # use this to fix a particular parameter value
    param_override = {
        "HAS_BIAS": 1,
    }

    if seed is not None:
        seed = int(seed)
        mase_runner(
            trace=True,
            module_param_list=[get_mxint_linear_config_random(seed, param_override)],
        )
    else:
        num_configs = int(os.getenv("NUM_CONFIGS", default=5))
        base_seed = random.randrange(sys.maxsize)
        mase_runner(
            trace=True,
            module_param_list=[
                get_mxint_linear_config_random(base_seed + i, param_override)
                for i in range(num_configs)
            ],
            jobs=min(num_configs, 10),
        )
        print(f"Test seeds: \n{[(i,base_seed+i) for i in range(num_configs)]}")


@pytest.mark.dev
def test_mxint_linear():
    mase_runner(
        trace=True,
        module_param_list=[
            get_mxint_linear_config(
                {
                    "HAS_BIAS": 1,
                    "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
                    "DATA_IN_0_TENSOR_SIZE_DIM_1": 10,
                    "DATA_IN_0_PARALLELISM_DIM_0": 2,
                    "WEIGHT_TENSOR_SIZE_DIM_0": 4,
                    "WEIGHT_TENSOR_SIZE_DIM_1": 4,
                    "WEIGHT_PARALLELISM_DIM_0": 2,
                    "WEIGHT_PARALLELISM_DIM_1": 2,
                    "BIAS_TENSOR_SIZE_DIM_0": 4,
                    "BIAS_PARALLELISM_DIM_0": 2,
                }
            ),
        ],
    )


if __name__ == "__main__":
    # test_mxint_linear()
    test_mxint_linear_full_random()
