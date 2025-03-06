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
    await cocotb.start(Clock(dut.clk, 3.2, units = "ns").start())

    # Reset the module
    dut.rst.value = 1

    # Wait for clock cycle to progress
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)

    dut.rst.value = 0

# Test with input: 4'b1010, expected output: 4'b1100
@cocotb.test()
async def test_input(dut: HierarchyObject) -> None:
    # initialize ip
    await init(dut)

    dut.data_in_0.value = 10 # 4'b1010

    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)

    # Check the expected output - should be 4'b1100
    assert int(dut.data_out_0.value) == 12, f"Expected 12, got {int(dut.data_out.value)}"

    # wait 3 cycles
    await ClockCycles(dut.clk, 3)

def run_tests(log_level: int = logging.INFO, waves: bool = True):
    module = os.path.splitext(os.path.basename(__file__))[0]
    logging.getLogger('cocotb').setLevel(log_level)
    simulator.clean()
    
    return simulator.run(toplevel="fixed_difflogic_logic", module=module, waves=waves)
if __name__ == "__main__":
    run_tests()
