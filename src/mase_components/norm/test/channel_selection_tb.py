#!/usr/bin/env python3

from copy import copy
from pathlib import Path
from os import makedirs

# from itertools import batched  # Python 3.12

import torch
import cocotb
from cocotb.triggers import *

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench


class ChannelSelectionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        self.assign_self_params(["NUM_CHANNELS", "NUM_SPATIAL_BLOCKS", "S_STATE_WIDTH"])

    def generate_inputs(self, num_clocks):
        return num_clocks

    def model(self, num_clocks):
        model_out = []
        state = 0
        if self.NUM_CHANNELS < 2:
            for clock in range(num_clocks):
                print("ZEros")
                model_out.append(0)
        else:
            for clock in range(num_clocks):
                state = state % (self.MAX_STATE + 1)
                channel = state >> (self.STATE_WIDTH - self.OUT_WIDTH)
                model_out.append(channel)
                state += 1

        print("Model: ", model_out)

        return model_out


@cocotb.test()
async def basic(dut):
    tb = ChannelSelectionTB(dut)
    num_clocks = tb.generate_inputs(2 ** (tb.STATE_WIDTH + 1))
    ref = tb.model(num_clocks)

    await tb.reset()

    assert ref[0] == dut.channel.value.integer, f"<<< --- Test Failed --- >>>"

    for i in range(1, num_clocks):
        await RisingEdge(dut.clk)
        assert ref[i] == dut.channel.value.integer, f"<<< --- Test failed --- >>>"


import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_channel_selection():

    def gen_cfg(num_channels, num_blocks):
        return {"NUM_CHANNELS": num_channels, "NUM_SPATIAL_BLOCKS": num_blocks}

    def full_sweep():
        module_param_list = []
        for n in range(1, 3):
            for i in range(1, 8):
                params = gen_cfg(i, n)
                module_param_list.append(params)
        return module_param_list

    mase_runner(
        module_param_list=full_sweep(),
        # module_param_list=[gen_cfg(2, 1)],
        trace=True,
    )


if __name__ == "__main__":
    test_channel_selection()
