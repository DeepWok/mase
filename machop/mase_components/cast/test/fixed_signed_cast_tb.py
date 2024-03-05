import logging
from math import trunc, floor

import torch
import cocotb
from cocotb.triggers import *

from random import randint
from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import sign_extend_t

from chop.passes.graph.transforms.quantize.quantized_modules.fixed_signed_cast import (
    _fixed_signed_cast_model
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class FixedSignedCastTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)

        self.assign_self_params([
            "IN_WIDTH", "IN_FRAC_WIDTH", "OUT_WIDTH", "OUT_FRAC_WIDTH",
            "SYMMETRIC", "ROUND_FLOOR", "ROUND_TRUNCATE",
            "ROUND_NEAREST_INT_HALF_EVEN"
        ])

    def generate_inputs(self):
        uints = torch.arange(2**self.IN_WIDTH)
        num_int = sign_extend_t(uints, self.IN_WIDTH)
        num_float = num_int / (2**self.IN_FRAC_WIDTH)
        return num_int, num_float

    def rounding_mode(self):
        if self.ROUND_FLOOR:
            return "floor"
        elif self.ROUND_TRUNCATE:
            return "trunc"
        elif self.ROUND_NEAREST_INT_HALF_EVEN:
            return "round_nearest_half_even"
        else:
            raise Exception("Rounding mode not recognised.")

    def model(self, inputs):
        return _fixed_signed_cast_model(
            float_input=inputs,
            out_width=self.OUT_WIDTH,
            out_frac_width=self.OUT_FRAC_WIDTH,
            symmetric=self.SYMMETRIC,
            rounding_mode=self.rounding_mode()
        )


@cocotb.test()
async def exhaustive_test(dut):
    tb = FixedSignedCastTB(dut)
    driver_in, float_in = tb.generate_inputs()
    exp_float, exp_output = tb.model(float_in)

    for i in range(driver_in.shape[0]):
        x = driver_in[i].item()
        exp_y = exp_output[i].item()

        dut.in_data.value = x
        await Timer(10, "ns")
        got_y = int(dut.out_data.value)

        assert got_y == exp_output[i], (
            f"Output did not match! Got {got_y}, Exp {exp_y}"
        )


if __name__ == "__main__":
    DEFAULT_CONFIG = {
        "IN_WIDTH": 8,
        "IN_FRAC_WIDTH": 2,
        "OUT_WIDTH": 8,
        "OUT_FRAC_WIDTH": 2,
        "SYMMETRIC": 0,
        "ROUND_FLOOR": 1,
        "ROUND_TRUNCATE": 0,
        "ROUND_NEAREST_INT_HALF_EVEN": 0
    }

    def gen_width_change_configs(cfg_list):
        l = list()
        for cfg in cfg_list:
            l.extend([
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1
                },
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1
                },
                {
                    **cfg,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"]+1
                },
                {
                    **cfg,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"]-1
                },
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] - 2
                },
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] + 2
                },
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] - 2
                },
                {
                    **cfg,
                    "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1,
                    "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] + 2
                },
            ])
        return l

    def gen_symmetric(cfg_list):
        l = list()
        for cfg in cfg_list:
            l.extend([
                {**cfg, "SYMMETRIC": 0},
                {**cfg, "SYMMETRIC": 1},
            ])
        return l

    def gen_rounding(cfg_list):
        l = list()
        for cfg in cfg_list:
            l.extend([
                {**cfg,
                    "ROUND_FLOOR": 1,
                    "ROUND_TRUNCATE": 0,
                    "ROUND_NEAREST_INT_HALF_EVEN": 0
                },
                {**cfg,
                    "ROUND_FLOOR": 0,
                    "ROUND_TRUNCATE": 1,
                    "ROUND_NEAREST_INT_HALF_EVEN": 0
                },
                {**cfg,
                    "ROUND_FLOOR": 0,
                    "ROUND_TRUNCATE": 0,
                    "ROUND_NEAREST_INT_HALF_EVEN": 1
                },
            ])
        return l

    cfg_list = [DEFAULT_CONFIG]
    cfg_list = gen_width_change_configs(cfg_list)
    cfg_list = gen_symmetric(cfg_list)
    # Other rounding modes not supported yet
    # cfg_list = gen_rounding(cfg_list)

    mase_runner(
        module_param_list=[
            DEFAULT_CONFIG,
            *cfg_list,
            {
                **DEFAULT_CONFIG,
                "IN_WIDTH": 10,
                "IN_FRAC_WIDTH": 2,
                "OUT_WIDTH": 8,
                "OUT_FRAC_WIDTH": 1,
            }
        ],
        trace=True,
    )
