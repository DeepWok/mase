# TB for difflogic_flatten which takes in 2D array of binary vals and makes into flattened 1D array
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
async def test_flatten(dut: HierarchyObject) -> None:
    # initialize ip
    await init(dut)

    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1

    # Create 2x2 tensor:
    # [ 1 0 ]
    # [ 0 1 ]
    dut.data_in_0[0].value = 2  # 2'b10
    dut.data_in_0[1].value = 1  # 2'b01

    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)

    # Flattened output:
    # [ 0 1 1 0 ] - Expected output = 6
    assert (
        int(dut.data_out_0.value) == 6
    ), f"Expected 6, got {int(dut.data_out_0.value)}"  # 4'b0110


def run_tests(log_level: int = logging.INFO, waves: bool = True):
    module = os.path.splitext(os.path.basename(__file__))[0]
    logging.getLogger("cocotb").setLevel(log_level)
    simulator.clean()

    return simulator.run(toplevel="fixed_difflogic_flatten", module=module, waves=waves)


if __name__ == "__main__":
    run_tests()
