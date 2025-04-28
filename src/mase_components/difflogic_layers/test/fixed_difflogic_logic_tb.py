# TB for fixed_difflogic_logic (difflogic_layer)
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
async def test_difflogic_logic(dut: HierarchyObject) -> None:
    # initialize ip
    await init(dut)

    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1

    # Pass binary input:
    # 4'b1010
    dut.data_in_0.value = 10

    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)

    # Difflogic_logic output:
    # Exoected output = 4'b1100
    assert (
        int(dut.data_out_0.value) == 12
    ), f"Expected 12, got {int(dut.data_out.value)}"


def run_tests(log_level: int = logging.INFO, waves: bool = True):
    module = os.path.splitext(os.path.basename(__file__))[0]
    logging.getLogger("cocotb").setLevel(log_level)
    simulator.clean()

    return simulator.run(toplevel="fixed_difflogic_logic", module=module, waves=waves)


if __name__ == "__main__":
    run_tests()
