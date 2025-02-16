import logging
from math import trunc, floor

import torch
import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.runner import mase_runner


# def _fixed_signed_cast_model(
#     float_input, out_width, out_frac_width, symmetric, rounding_mode
# ):
#     scaled_float = float_input * (2**out_frac_width)
#     if rounding_mode == "floor":
#         out_int = my_floor(scaled_float)
#     elif rounding_mode == "round_nearest_half_even":
#         out_int = my_round(scaled_float)
#     else:
#         raise Exception("Rounding mode not recognised.")
#     out_int = my_clamp(
#         out_int,
#         -(2 ** (out_width - 1)) + 1 if symmetric else -(2 ** (out_width - 1)),
#         (2 ** (out_width - 1)) - 1,
#     )
#     out_float = out_int / (2**out_frac_width)
#     # out_uint is a non-differentiable path
#     out_uint = signed_to_unsigned(out_int.int(), out_width)
#     return out_float, out_uint


logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class FixedUnsignedCastTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut)

        self.assign_self_params(
            [
                "IN_WIDTH",
                "IN_FRAC_WIDTH",
                "OUT_WIDTH",
                "OUT_FRAC_WIDTH",
                "ROUND_FLOOR",
                "ROUND_TRUNCATE",
                "ROUND_NEAREST_INT_HALF_EVEN",
            ]
        )

    def generate_inputs(self):
        return torch.arange(2**self.IN_WIDTH)

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
        float_input = inputs / (2**self.IN_FRAC_WIDTH)
        scaled_float = float_input * (2**self.OUT_FRAC_WIDTH)
        rounded = torch.floor(scaled_float)
        model_out = torch.clamp(rounded, 0, (2**self.OUT_WIDTH - 1))
        return model_out

    async def run_test(self):
        inputs = self.generate_inputs()
        exp_output = self.model(inputs)

        logger.info(inputs)
        logger.info(exp_output)

        for i in range(inputs.shape[0]):
            x = inputs[i].item()
            exp_y = exp_output[i].item()

            self.dut.in_data.value = x
            await Timer(10, "ns")
            got_y = int(self.dut.out_data.value)

            assert (
                got_y == exp_output[i]
            ), f"Output did not match! Got {got_y}, Exp {exp_y}"


@cocotb.test()
async def sweep(dut):
    tb = FixedUnsignedCastTB(dut)
    await tb.run_test()


if __name__ == "__main__":
    DEFAULT_CONFIG = {
        "IN_WIDTH": 8,
        "IN_FRAC_WIDTH": 2,
        "OUT_WIDTH": 8,
        "OUT_FRAC_WIDTH": 2,
        "ROUND_FLOOR": 1,
        "ROUND_TRUNCATE": 0,
        "ROUND_NEAREST_INT_HALF_EVEN": 0,
    }

    def gen_width_change_configs(cfg_list):
        l = list()
        for cfg in cfg_list:
            l.extend(
                [
                    {**cfg, "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1},
                    {**cfg, "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1},
                    {**cfg, "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] + 1},
                    {**cfg, "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] - 1},
                    {
                        **cfg,
                        "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1,
                        "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] - 2,
                    },
                    {
                        **cfg,
                        "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] + 1,
                        "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] + 2,
                    },
                    {
                        **cfg,
                        "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1,
                        "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] - 2,
                    },
                    {
                        **cfg,
                        "OUT_WIDTH": DEFAULT_CONFIG["OUT_WIDTH"] - 1,
                        "OUT_FRAC_WIDTH": DEFAULT_CONFIG["OUT_FRAC_WIDTH"] + 2,
                    },
                ]
            )
        return l

    def gen_rounding(cfg_list):
        l = list()
        for cfg in cfg_list:
            l.extend(
                [
                    {
                        **cfg,
                        "ROUND_FLOOR": 1,
                        "ROUND_TRUNCATE": 0,
                        "ROUND_NEAREST_INT_HALF_EVEN": 0,
                    },
                    {
                        **cfg,
                        "ROUND_FLOOR": 0,
                        "ROUND_TRUNCATE": 1,
                        "ROUND_NEAREST_INT_HALF_EVEN": 0,
                    },
                    {
                        **cfg,
                        "ROUND_FLOOR": 0,
                        "ROUND_TRUNCATE": 0,
                        "ROUND_NEAREST_INT_HALF_EVEN": 1,
                    },
                ]
            )
        return l

    cfg_list = [DEFAULT_CONFIG]
    cfg_list = gen_width_change_configs(cfg_list)
    # Other rounding modes not supported yet
    # cfg_list = gen_rounding(cfg_list)

    mase_runner(
        module_param_list=[
            # DEFAULT_CONFIG,
            *cfg_list,
            # {
            #     **DEFAULT_CONFIG,
            #     "IN_WIDTH": 10,
            #     "IN_FRAC_WIDTH": 2,
            #     "OUT_WIDTH": 8,
            #     "OUT_FRAC_WIDTH": 1,
            # },
        ],
        trace=True,
    )
