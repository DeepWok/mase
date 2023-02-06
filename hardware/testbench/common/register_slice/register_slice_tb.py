# This script tests the register slice
import random, os

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner


@cocotb.test()
async def test_register_slice(dut):
    """ Test the handshake interface and add random number inputs """

    # Reset cycle
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Test inputs outputs
    # The test bench test two features: the correctness of values and
    # the correctness of control logic. In order to test the correctness
    # of data, randome values are input and expected to be passed to
    # the output. In oder to test the correctness of the control logic,
    # the states of handshake signals are enumerated to verify all the
    # possible cases.

    for i in range(30):

        # Clear data path
        dut.w_valid.value = 0
        dut.r_ready.value = 1
        # Synchronize with the clock
        await FallingEdge(dut.clk)
        await FallingEdge(dut.clk)
        expected_w_ready = 1
        expected_r_valid = 0
        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (init state) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (init state) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # Four different cases are covered:
        # 1. w_valid and r_ready => no backpressure
        # 1.1 empty data path => pass input
        data_in = random.randint(0, 15)
        dut.w_data.value = data_in
        dut.w_valid.value = 1
        dut.r_ready.value = 1
        await FallingEdge(dut.clk)
        expected_r_data = data_in
        expected_w_ready = 1
        expected_r_valid = 1

        assert dut.r_data.value == expected_r_data, "Handshake check for case (w_valid and r_ready, empty data path) failed. Output data = {}, expect: {}".format(
            dut.r_data.value, expected_r_data)
        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (w_valid and r_ready, empty data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (w_valid and r_ready, empty data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # 1.2 loaded data path => pass input
        data_in = random.randint(0, 15)
        dut.w_data.value = data_in
        dut.w_valid.value = 1
        dut.r_ready.value = 1
        await FallingEdge(dut.clk)
        expected_r_data = data_in
        expected_w_ready = 1
        expected_r_valid = 1

        assert dut.r_data.value == expected_r_data, "Handshake check for case (w_valid and r_ready, loaded data path) failed. Output data = {}, expect: {}".format(
            dut.r_data.value, expected_r_data)
        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (w_valid and r_ready, loaded data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (w_valid and r_ready, loaded data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # 2. w_valid and !r_ready => back pressure
        # 2.1 loaded data path => block input
        new_data_in = data_in + 1
        dut.w_data.value = new_data_in
        dut.w_valid.value = 1
        dut.r_ready.value = 0
        await FallingEdge(dut.clk)
        expected_r_data = data_in
        expected_w_ready = 0
        expected_r_valid = 1

        assert dut.r_data.value == expected_r_data, "Handshake check for case (w_valid and !r_ready, loaded data path) failed. Output data = {}, expect: {}".format(
            dut.r_data.value, expected_r_data)
        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (w_valid and !r_ready, loaded data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (w_valid and !r_ready, loaded data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # Clear data path
        data_in = random.randint(0, 15)
        dut.w_data.value = data_in
        dut.w_valid.value = 0
        dut.r_ready.value = 1
        await FallingEdge(dut.clk)

        # 2.2 empty data path => pass input
        data_in = random.randint(0, 15)
        dut.w_data.value = data_in
        dut.w_valid.value = 1
        dut.r_ready.value = 0
        await FallingEdge(dut.clk)
        expected_r_data = data_in
        expected_w_ready = 0
        expected_r_valid = 1

        assert dut.r_data.value == expected_r_data, "Handshake check for case (w_valid and !r_ready, empty data path) failed. Output data = {}, expect: {}".format(
            dut.r_data.value, expected_r_data)
        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (w_valid and !r_ready, empty data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (w_valid and !r_ready, empty data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # 3. !w_valid and r_ready => no backpressure
        # 3.1 loaded data path => offload output
        dut.w_valid.value = 0
        dut.r_ready.value = 1
        await FallingEdge(dut.clk)
        expected_w_ready = 1
        expected_r_valid = 0

        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (!w_valid and r_ready, loaded data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (!w_valid and r_ready, loaded data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # 3.2 empty data path => idle
        # Skipped because it is the same as init state

        # 4. !w_valid and !r_ready => no backpressure
        # 3.1 empty data path => idle
        dut.w_valid.value = 0
        dut.r_ready.value = 0
        await FallingEdge(dut.clk)
        expected_w_ready = 1
        expected_r_valid = 0

        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (!w_valid and !r_ready, empty data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (!w_valid and !r_ready, empty data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)

        # Load datapath
        dut.w_valid.value = 1
        dut.r_ready.value = 0
        await FallingEdge(dut.clk)

        # 3.2 loaded data path => idle
        dut.w_valid.value = 0
        dut.r_ready.value = 0
        await FallingEdge(dut.clk)
        expected_w_ready = 0
        expected_r_valid = 1

        assert dut.w_ready.value == expected_w_ready, "Handshake check for case (!w_valid and !r_ready, loaded data path) failed. w_ready = {}, expect: {}".format(
            dut.w_ready.value, expected_w_ready)
        assert dut.r_valid.value == expected_r_valid, "Handshake check for case (!w_valid and !r_ready, loaded data path) failed. r_valid = {}, expect: {}".format(
            dut.r_valid.value, expected_r_valid)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/common/register_slice.sv",
    ]

    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources, toplevel="register_slice")
    runner.test(toplevel="register_slice", py_module="register_slice_tb")


if __name__ == "__main__":
    runner()
