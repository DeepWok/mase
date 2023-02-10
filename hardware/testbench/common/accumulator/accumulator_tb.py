# This script tests the accumulator
import random, os, math

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner


class VerificationCase:

    def __init__(self, samples=10):
        self.in_width = 32
        self.num = 9
        self.out_width = math.ceil(math.log2(self.num)) + self.in_width
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i = [random.randint(0, 30) for _ in range(self.num)]
            self.inputs += i
            self.outputs.append(sum(i))

    def get_dut_parameters(self):
        return {
            'NUM': self.num,
            'IN_WIDTH': self.in_width,
            'OUT_WIDTH': self.out_width,
        }

    def get_dut_input(self, i):
        return self.inputs[i]

    def get_dut_num(self):
        return self.num

    def get_dut_output(self, i):
        return self.outputs[i]


@cocotb.test()
async def test_accumulator(dut):
    """ Test accumulator by randomly sending inputs """
    samples = 20
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    dut.ind_valid.value = 0
    dut.outd_ready.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    iterations = test_case.get_dut_num()
    output_count = samples
    input_count = output_count * iterations
    # Synchronize with the clock
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)

    # We first test whether the logic for handling back pressure
    # is correctly implemented. We inject a set of valid inputs
    # to fill the pipeline and disable the output ready signal
    # to see if the last output is kept and the input is blocked.

    # Test forward path without back pressure
    j = 0
    for i in range(2 * iterations):
        dut.ind_valid.value = 1
        dut.ind.value = test_case.get_dut_input(i)
        dut.outd_ready.value = 1
        await FallingEdge(dut.clk)
        notvalid = not dut.outd_valid.value
        to_output = ((i + 1) % iterations == 0)
        assert dut.ind_ready.value, 'Error: Input ready set at cycle {} without back pressure: {}'.format(
            i, int(dut.ind_ready.value))
        assert notvalid or to_output, 'Error: Output valid at cycle {} without back pressure: {}'.format(
            i, int(dut.outd_valid.value))
        assert notvalid or dut.outd.value == test_case.get_dut_output(
            j
        ), 'Error: Output mismatch at cycle {} without back pressure: {}, expected: {}'.format(
            i, int(dut.outd.value), int(test_case.get_dut_output(j)))
        j += ((i + 1) % iterations == 0)

    # Add a sudden back pressure
    assert dut.outd_valid.value, 'Error: Prepared output valid signal is clear'
    i = 2 * iterations
    dut.ind_valid.value = 1
    dut.ind.value = test_case.get_dut_input(i)
    dut.outd_ready.value = 0
    await FallingEdge(dut.clk)
    assert dut.outd_valid.value, 'Error: Output valid is clear when a sudden back pressure happens'
    assert not dut.ind_ready.value, 'Error: Input ready is set when a sudden back pressure happens'

    # Test forward path without back pressure
    for i in range(2 * iterations, 4 * iterations):
        dut.ind_valid.value = 1
        dut.ind.value = test_case.get_dut_input(i)
        dut.outd_ready.value = 1
        await FallingEdge(dut.clk)
        notvalid = not dut.outd_valid.value
        to_output = ((i + 1) % iterations == 0)
        assert dut.ind_ready.value, 'Error: Input ready set at cycle {} without back pressure: {}'.format(
            i, int(dut.ind_ready.value))
        assert notvalid or to_output, 'Error: Output valid at cycle {} without back pressure: {}'.format(
            i, int(dut.outd_valid.value))
        assert notvalid or dut.outd.value == test_case.get_dut_output(
            j
        ), 'Error: Output mismatch at cycle {} without back pressure: {}, expected: {}'.format(
            i, int(dut.outd.value), int(test_case.get_dut_output(j)))
        j += ((i + 1) % iterations == 0)

    # We then test whether the logic for handling input handshake
    # is correctly implemented. We inject a set of inputs with bubbles
    # to see if the output is still correct.

    for i in range(4 * iterations, input_count):
        # Inject a bubble
        dut.ind_valid.value = 0
        dut.ind.value = random.randint(0, 15)
        dut.outd_ready.value = 1
        await FallingEdge(dut.clk)
        if dut.outd_valid.value:
            assert dut.outd.value == test_case.get_dut_output(
                j
            ), 'Error: Output mismatch at cycle {} without back pressure: {}, expected: {}'.format(
                i, int(dut.outd.value), int(test_case.get_dut_output(j)))
        dut.ind_valid.value = 1
        dut.ind.value = test_case.get_dut_input(i)
        dut.outd_ready.value = 1
        await FallingEdge(dut.clk)
        if dut.outd_valid.value:
            assert dut.outd.value == test_case.get_dut_output(
                j
            ), 'Error: Output mismatch at cycle {} without back pressure: {}, expected: {}'.format(
                i, int(dut.outd.value), int(test_case.get_dut_output(j)))
        j += ((i + 1) % iterations == 0)
    assert j == output_count, 'Error: transaction mismatch: {}, expected: {}'.format(
        j, output_count)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = ["../../../../hardware/common/accumulator.sv"]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f'-G{k}={v}')
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(verilog_sources=verilog_sources,
                 toplevel="accumulator",
                 extra_args=extra_args)

    runner.test(toplevel="accumulator", py_module="accumulator_tb")


if __name__ == "__main__":
    runner()
