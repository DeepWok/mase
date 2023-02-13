#!/usr/bin/env python3

# This script tests the adder tree
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
        self.in_width = 32
        self.num = 8  # 13
        self.out_width = math.ceil(math.log2(self.num)) + 32
        self.inputs = RandomSource(samples=samples,
                                   num=self.num,
                                   max_stalls=2 * samples,
                                   debug=debug)
        self.outputs = RandomSink(samples=samples,
                                  num=self.num,
                                  max_stalls=2 * samples,
                                  debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            'NUM': self.num,
            'IN_WIDTH': self.in_width,
            'OUT_WIDTH': self.out_width,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            ref.append(sum(self.inputs.data[i]))
        ref.reverse()
        return ref


# Check if an impossible state is reached
def is_impossible_state(in_ready, in_valid, out_ready, out_valid):
    # (0, X, 0, 0)
    # (0, X, 1, 0)
    # (0, X, 1, 1)
    if (not in_ready) and not ((not out_ready) and out_valid):
        return True
    return False


@cocotb.test()
async def test_adder_tree(dut):
    """ Test integer based adder tree """
    samples = 20
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
    dut.in_valid.value = 0
    dut.out_ready.value = 1
    logger.debug(
        'Pre-clk  State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
        .format(dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
                dut.out_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
        .format(dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
                dut.out_valid.value))
    logger.debug(
        'Pre-clk  State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
        .format(dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
                dut.out_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
        .format(dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
                dut.out_valid.value))

    done = False
    while not done:
        await FallingEdge(dut.clk)
        logger.debug(
            'Post-clk State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
            .format(dut.in_ready.value, dut.in_valid.value,
                    dut.out_ready.value, dut.out_valid.value))
        assert not is_impossible_state(
            dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
            dut.out_valid.value
        ), 'Error: invalid state (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'.format(
            dut.in_ready.value, dut.in_valid.value, dut.out_ready.value,
            dut.out_valid.value)
        dut.in_valid.value, dut.ind.value = test_case.inputs.compute(
            dut.in_ready.value)
        dut.out_ready.value = test_case.outputs.compute(
            dut.out_valid.value, dut.outd.value)
        logger.debug(
            'Pre-clk  State: (in_ready,in_valid,out_ready,out_valid) = ({},{},{},{})'
            .format(dut.in_ready.value, dut.in_valid.value,
                    dut.out_ready.value, dut.out_valid.value))
        done = test_case.inputs.is_empty() and test_case.outputs.is_full()

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/common/adder_tree.sv",
        "../../../../hardware/common/register_slice.sv",
        "../../../../hardware/common/adder_tree_layer.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f'-G{k}={v}')
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources,
                 toplevel="adder_tree",
                 extra_args=extra_args)

    runner.test(toplevel="adder_tree", py_module="adder_tree_tb")


if __name__ == "__main__":
    runner()
