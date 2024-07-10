#!/usr/bin/env python3

import os, logging, pytest
from random import randint
from pathlib import Path

from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import verilator_str_param

import cocotb
from cocotb.triggers import *

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class LutTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk)
        self.assign_self_params(["DATA_WIDTH", "SIZE", "OUTPUT_REG"])

    def generate_inputs(self, batches=1):
        """Generate random addresses."""
        return [randint(0, (self.SIZE - 1)) for _ in range(batches)]


@cocotb.test()
async def exhaust(dut):
    tb = LutTB(dut)
    if tb.OUTPUT_REG:
        # Clocked module
        for addr in range(tb.SIZE):
            await RisingEdge(dut.clk)
            dut.addr.value = addr
            await RisingEdge(dut.clk)
            await ReadOnly()
            data_out = int(dut.data.value)

            if addr != data_out:
                logger.error("Failed %d != %d" % (addr, data_out))
                assert False
            else:
                logger.info("Passed %d == %d" % (addr, data_out))

    else:
        # Combinatorial module
        for addr in range(tb.SIZE):
            dut.addr.value = addr
            await Timer(10, units="ns")
            data_out = int(dut.data.value)

            if addr != data_out:
                logger.error("Failed %d != %d" % (addr, data_out))
                assert False
            else:
                logger.info("Passed %d == %d" % (addr, data_out))


def write_memb(file: Path, nums: list[int], width: int):
    num_str = ["{0:0{width}b}".format(x, width=width) for x in nums]
    with open(file, "w") as f:
        #  !!! Need to write new line at end of file or $readmem will silently
        #  fail on last line of mem file :(
        f.write("\n".join(num_str) + "\n")


def read_memb(file: Path, width):
    with open(file, "r") as f:
        lines = f.readlines()
    return [int(line.strip(), 2) for line in lines]


@pytest.mark.dev
def test_lut():
    DATA_WIDTH = 8
    SIZE = 13
    lut = list(range(SIZE))
    memfile = Path(__file__).parent.parent / "rtl" / "lut_mem0.mem"
    write_memb(memfile, lut, width=DATA_WIDTH)
    config = {
        "DATA_WIDTH": DATA_WIDTH,
        "SIZE": SIZE,
        "OUTPUT_REG": 1,
        "MEM_FILE": verilator_str_param(str(memfile)),
    }
    mase_runner(module_param_list=[config], trace=True)


if __name__ == "__main__":
    test_lut()
