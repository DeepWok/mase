import cocotb
from cocotb.triggers import *
from cocotb.clock import Clock


class Testbench:
    def __init__(self, dut, clk=None, rst=None) -> None:
        self.dut = dut
        self.clk = clk
        self.rst = rst

        if self.clk != None:
            self.clock = Clock(self.clk, 20, units="ns")
            cocotb.start_soon(self.clock.start())

    def assign_self_params(self, attrs):
        for att in attrs:
            setattr(self, att, getattr(self.dut, att).value)

    async def reset(self, active_high=True):
        if self.rst == None:
            raise Exception(
                "Cannot reset. Either a reset wire was not provided or "
                + "the module does not have a reset."
            )

        await RisingEdge(self.clk)
        self.rst.value = 1 if active_high else 0
        await RisingEdge(self.clk)
        self.rst.value = 0 if active_high else 1
        await RisingEdge(self.clk)
