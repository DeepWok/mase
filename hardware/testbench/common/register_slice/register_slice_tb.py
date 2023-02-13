#!/usr/bin/env python3

# This script tests the register slice
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
print(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from RandomTest import RandomSource
from RandomTest import RandomSink
from RandomTest import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

debug = False

logger = logging.getLogger('tb_signals')
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:

    def __init__(self, samples=10):
        self.data_width = 32
        self.inputs = RandomSource(samples=samples,
                                   max_stalls=2 * samples,
                                   is_data_vector=False,
                                   debug=debug)
        self.outputs = RandomSink(samples=samples,
                                  max_stalls=2 * samples,
                                  debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            'DATA_WIDTH': self.data_width,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(self.inputs.data[i])
        ref.reverse()
        return ref


# Check if an impossible state is reached
def impossiblestate(w_ready, w_valid, r_ready, r_valid):
    # (0, X, 0, 0)
    # (0, X, 1, 0)
    # (0, X, 1, 1)
    if (not w_ready) and not ((not r_ready) and r_valid):
        return True


@cocotb.test()
async def test_register_slice(dut):
    """ Test register slice """
    samples = 30
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Synchronize with the clock
    dut.w_valid.value = 0
    dut.r_ready.value = 1
    logger.debug(
        'Pre-clk  State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'.
        format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
               dut.r_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'.
        format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
               dut.r_valid.value))
    logger.debug(
        'Pre-clk  State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'.
        format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
               dut.r_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'.
        format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
               dut.r_valid.value))

    done = False
    while not done:
        await FallingEdge(dut.clk)
        logger.debug(
            'Post-clk State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
                    dut.r_valid.value))
        assert not impossiblestate(
            dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
            dut.r_valid.value
        ), 'Error: invalid state (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'.format(
            dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
            dut.r_valid.value)
        dut.w_valid.value, dut.w_data.value = test_case.inputs.compute(
            dut.w_ready.value)
        dut.r_ready.value = test_case.outputs.compute(dut.r_valid.value,
                                                      dut.r_data.value)
        logger.debug(
            'Pre-clk  State: (w_ready,w_valid,r_ready,r_valid) = ({},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.r_ready.value,
                    dut.r_valid.value))
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/common/register_slice.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f'-G{k}={v}')
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources,
                 toplevel="register_slice",
                 extra_args=extra_args)

    runner.test(toplevel="register_slice", py_module="register_slice_tb")


if __name__ == "__main__":
    runner()
