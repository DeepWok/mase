#!/usr/bin/env python3

# This script tests the dot product
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
        self.act_width = 32
        self.w_width = 16
        self.vector_size = 2
        self.register_levels = 1
        self.act = RandomSource(name='act',
                                samples=samples,
                                num=self.vector_size,
                                max_stalls=2 * samples,
                                debug=debug)
        self.w = RandomSource(name='w',
                              samples=samples,
                              num=self.vector_size,
                              max_stalls=2 * samples,
                              debug=debug)
        self.outputs = RandomSink(samples=samples,
                                  max_stalls=2 * samples,
                                  debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            'ACT_WIDTH': self.act_width,
            'W_WIDTH': self.w_width,
            'VECTOR_SIZE': self.vector_size,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            s = [
                self.act.data[i][j] * self.w.data[i][j]
                for j in range(self.vector_size)
            ]
            ref.append(sum(s))
        ref.reverse()
        return ref


# Check if an is_impossible state is reached
def is_impossible_state(w_ready, w_valid, act_ready, act_valid, out_ready,
                        out_valid):
    return False


@cocotb.test()
async def test_dot_product(dut):
    """ Test integer based vector mult """
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
    dut.w_valid.value = 0
    dut.act_valid.value = 0
    dut.out_ready.value = 1
    logger.debug(
        'Pre-clk  State: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
        .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                dut.act_valid.value, dut.out_ready.value, dut.out_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
        .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                dut.act_valid.value, dut.out_ready.value, dut.out_valid.value))
    logger.debug(
        'Pre-clk  State: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
        .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                dut.act_valid.value, dut.out_ready.value, dut.out_valid.value))
    await FallingEdge(dut.clk)
    logger.debug(
        'Post-clk State: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
        .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                dut.act_valid.value, dut.out_ready.value, dut.out_valid.value))

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        logger.debug(
            'Post-clk State: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                    dut.act_valid.value, dut.out_ready.value,
                    dut.out_valid.value))
        assert not is_impossible_state(
            dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
            dut.act_valid.value, dut.out_ready.value, dut.out_valid.value
        ), 'Error: invalid state (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'.format(
            dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
            dut.act_valid.value, dut.out_ready.value, dut.out_valid.value)

        dut.w_valid.value = test_case.w.pre_compute(dut.w_ready.value)
        dut.act_valid.value = test_case.act.pre_compute(dut.act_ready.value)
        logger.debug(
            'Pre-clk State0: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                    dut.act_valid.value, dut.out_ready.value,
                    dut.out_valid.value))
        await Timer(1, units="ns")
        logger.debug(
            'Pre-clk State1: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                    dut.act_valid.value, dut.out_ready.value,
                    dut.out_valid.value))

        dut.w_valid.value, dut.weights.value = test_case.w.compute(
            dut.w_ready.value)
        dut.act_valid.value, dut.act.value = test_case.act.compute(
            dut.act_ready.value)
        dut.out_ready.value = test_case.outputs.compute(
            dut.out_valid.value, dut.outd.value)
        logger.debug(
            'Pre-clk State2: (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'
            .format(dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
                    dut.act_valid.value, dut.out_ready.value,
                    dut.out_valid.value))

        if test_case.w.is_empty() and test_case.act.is_empty(
        ) and test_case.outputs.is_full():
            done = True
            break
    assert done, 'Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)'

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/common/dot_product.sv",
        "../../../../hardware/common/vector_mult.sv",
        "../../../../hardware/common/register_slice.sv",
        "../../../../hardware/common/adder_tree.sv",
        "../../../../hardware/common/adder_tree_layer.sv",
        "../../../../hardware/common/int_mult.sv",
        "../../../../hardware/common/join2.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f'-G{k}={v}')
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources,
                 toplevel="dot_product",
                 extra_args=extra_args)

    runner.test(toplevel="dot_product", py_module="dot_product_tb")


if __name__ == "__main__":
    runner()
