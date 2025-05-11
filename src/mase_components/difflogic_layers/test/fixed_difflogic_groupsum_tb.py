# TB for difflogic_groupsum which takes in array of binary vals and counts number of ones every group (2 elements)
import os
import logging

import cocotb
import cocotb_test.simulator as simulator
from cocotb.clock import Clock
from cocotb.handle import HierarchyObject
from cocotb.triggers import RisingEdge, ClockCycles


async def init(dut: HierarchyObject) -> None:
    # Start clk
    await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0


@cocotb.test()
async def test_groupsum(dut: HierarchyObject) -> None:
    # initialize ip
    await init(dut)

    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1

    # Create array of binary vals:
    dut.data_in_0.value = 11  # 4'b1011 == 11

    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)

    # Count ones of element pairs:
    # Expected data_out_0: count_ones(10) = 1, count_ones(11) = 2 so [[2'b01], [2'b11]]
    assert (
        int(dut.data_out_0[0].value) == 2
    ), f"Expected 2, got {int(dut.data_out_0[0].value)}"  # 2'b01
    assert (
        int(dut.data_out_0[1].value) == 1
    ), f"Expected 1, got {int(dut.data_out_0[1].value)}"  # 2'b11


def run_tests(log_level: int = logging.INFO, waves: bool = True):
    module = os.path.splitext(os.path.basename(__file__))[0]
    logging.getLogger("cocotb").setLevel(log_level)
    simulator.clean()

    return simulator.run(
        toplevel="fixed_difflogic_groupsum", module=module, waves=waves
    )


if __name__ == "__main__":
    run_tests()
