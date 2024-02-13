import logging
from math import trunc, floor

import cocotb
from cocotb.triggers import *

from random import randint
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import sign_extend, sign_extend_t, signed_to_unsigned

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def get_input(uint, in_width, in_frac_width):
    num_int = sign_extend(uint, in_width)
    num_float = float(num_int) / (2**in_frac_width)

    logger.debug("Drive bus with %d, signed: %d, float: %f"
                  % (uint, num_int, num_float))
    return uint, num_float

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def get_output(float_input, out_width, out_frac_width, symmetric, rounding_mode):
    if rounding_mode == "floor":
        out_int = floor(float_input * (2 ** out_frac_width))
    elif rounding_mode == "trunc":
        out_int = trunc(float_input * (2 ** out_frac_width))
    elif rounding_mode == "round_nearest_half_even":
        # Note: Python round is nearest int, round half to even
        out_int = round(float_input * (2 ** out_frac_width))
    else:
        raise Exception("Rounding mode not recognised.")
    out_float = out_int / (2 ** out_frac_width)
    out_int = clamp(out_int,
        smallest=-(2**(out_width-1))+1 if symmetric else -(2**(out_width-1)),
        largest=(2**(out_width-1))-1
    )
    out_uint = signed_to_unsigned(out_int, out_width)
    return out_uint, out_float

def get_rounding_mode(dut):
    if int(dut.ROUND_FLOOR.value):
        return "floor"
    elif int(dut.ROUND_TRUNCATE.value):
        return "trunc"
    elif int(dut.ROUND_NEAREST_INT_HALF_EVEN.value):
        return "round_nearest_half_even"
    else:
        raise Exception("Rounding mode not recognised.")

@cocotb.test()
async def exhaustive_test(dut):
    IN_WIDTH = int(dut.IN_WIDTH.value)
    IN_FRAC_WIDTH = int(dut.IN_FRAC_WIDTH.value)
    OUT_WIDTH = int(dut.OUT_WIDTH.value)
    OUT_FRAC_WIDTH = int(dut.OUT_FRAC_WIDTH.value)
    SYMMETRIC = int(dut.SYMMETRIC.value)
    await Timer(10, "ns")

    logger.debug(f"MIN_VAL: {dut.clamp_inst.MIN_VAL.value}")
    logger.debug(f"MAX_VAL: {dut.clamp_inst.MAX_VAL.value}")

    for uint in range(2**IN_WIDTH):
        x, x_float = get_input(uint, IN_WIDTH, IN_FRAC_WIDTH)

        dut.in_data.value = x
        await Timer(10, "ns")
        y = int(dut.out_data.value)

        signed_x = sign_extend(x, IN_WIDTH)
        got_signed_y = sign_extend(y, OUT_WIDTH)
        got_float_y = got_signed_y / (2**OUT_FRAC_WIDTH)

        y_exp, y_float = get_output(x_float, OUT_WIDTH, OUT_FRAC_WIDTH,
                                    SYMMETRIC, get_rounding_mode(dut))

        exp_signed_y = sign_extend(y_exp, OUT_WIDTH)
        exp_float_y = exp_signed_y / (2**OUT_FRAC_WIDTH)
        # logger.debug(f"out_int: {dut.floor_round_inst.out_int.value}, out_frac: {dut.floor_round_inst.out_frac.value}")
        logger.debug(f"round_out: {dut.round_out.value}")
        logger.debug(f"clamp_bounds: {dut.clamp_inst.MIN_VAL.value} <-> {dut.clamp_inst.MAX_VAL.value}")
        logger.debug(f"clamp: {dut.clamp_inst.in_data.value} -> {dut.clamp_inst.out_data.value}")
        logger.debug(f"GOT: 0x{x:x} [{signed_x}] ({x_float}) " +
                     f"-> 0x{y:x} [{got_signed_y}] ({got_float_y})")
        logger.debug(f"EXP: 0x{x:x} [{signed_x}] ({x_float}) " +
                     f"-> 0x{y_exp:x} [{exp_signed_y}] ({exp_float_y})")

        assert y == y_exp, "Output doesn't match with expected."

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
