#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

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
from utils import MxIntCast

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class MxIntCastTB(Testbench):
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
        )
        self.input_drivers = {"in0": self.data_in_0_driver}
        self.output_monitors = {"out": self.data_out_0_monitor}
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self):
        inputs = []
        exp_outputs = []
        for _ in range(self.num):
            data = 20 * torch.rand(int(self.dut.BLOCK_SIZE))
            (data_in, mdata_in, edata_in) = mxint_quantize(
                data,
                int(self.dut.IN_MAN_WIDTH),
                int(self.dut.IN_EXP_WIDTH),
            )
            mexp_out, eexp_out = MxIntCast(
                mdata_in,
                edata_in,
                {
                    "in_width": int(self.dut.IN_MAN_WIDTH),
                    "in_frac_width": int(self.dut.IN_MAN_FRAC_WIDTH),
                    "in_exponent_width": int(self.dut.IN_EXP_WIDTH),
                    "out_width": int(self.dut.OUT_MAN_WIDTH),
                    "out_exponent_width": int(self.dut.OUT_EXP_WIDTH),
                },
            )
            inputs.append((mdata_in.int().tolist(), edata_in.int().tolist()))
            exp_outputs.append((mexp_out.int().tolist(), eexp_out.int().tolist()))
        return inputs, exp_outputs

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")

        logger.info(f"generating inputs")
        inputs, exp_outputs = self.generate_inputs()

        # Load the inputs driver
        self.data_in_0_driver.load_driver(inputs)

        # Load the output monitor
        self.data_out_0_monitor.load_monitor(exp_outputs)
        await Timer(1, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    cocotb.start_soon(check_signal(dut))
    tb = MxIntCastTB(dut, num=1)
    await tb.run_test()


async def check_signal(dut):
    num = {"data_out_0": 0, "data_in_0": 0}
    await Timer(40, units="ns")
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.data_out_valid.value == 1 and dut.data_out_ready.value == 1:
            print(dut.edata_out_full)
            print(dut.log2_max_value)
            print(dut.ebuffer_data_for_out)
            shift = dut.ovshift_inst
            print(shift.SHIFT_WIDTH.value)
            print(shift.OUT_WIDTH.value)
            print(shift.shift_value.value.signed_integer)
            print(shift.abs_shift_value.value.signed_integer)
            # print(shift.shift_data.value.signed_integer)
            print([x for x in shift.shift_data.value])
        # print(dut.data_for_max_ready.value)
        # print(dut.data_for_out_valid.value)
        # print(dut.data_for_out_ready.value)
        print("end")
        # print(dut.max_bas_i.or_tree_i.gen_adder_tree.level[0].register_slice.data_out_ready)
        # print(dut.max_bas_i.or_tree_i.gen_adder_tree.level[0].register_slice.data_in_valid)
        # print(dut.max_bas_i.or_tree_i.gen_adder_tree.level[0].register_slice.shift_reg)
        # print(dut.max_bas_i.or_tree_i.data_in_ready)
        # print(dut.max_bas_i.data_out_ready)
        # print(dut.store_the_data.ff_inst.data_in_ready)
        # print(dut.store_the_data.ff_inst.data_out_ready)
        # print(dut.max_bas_i.or_tree_i.data_out_valid)
        # print("end")


if __name__ == "__main__":
    mase_runner(
        trace=True,
        module_param_list=[
            # {
            #     "IN_MAN_WIDTH": 6,
            #     "IN_MAN_FRAC_WIDTH": 5,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 12,
            #     "OUT_EXP_WIDTH": 4,
            #     "BLOCK_SIZE": 4,
            # },
            # {
            #     "IN_MAN_WIDTH": 8,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 3,
            #     "BLOCK_SIZE": 4,
            # },
            {
                "IN_MAN_WIDTH": 8,
                "IN_MAN_FRAC_WIDTH": 7,
                "IN_EXP_WIDTH": 4,
                "OUT_MAN_WIDTH": 16,
                "OUT_EXP_WIDTH": 5,
                "BLOCK_SIZE": 1,
            },
            # {
            #     "IN_MAN_WIDTH": 12,
            #     "IN_EXP_WIDTH": 3,
            #     "OUT_MAN_WIDTH": 8,
            #     "OUT_EXP_WIDTH": 4,
            #     "BLOCK_SIZE": 4,
            # },
        ],
        # sim="questa",
        # gui=True
    )
