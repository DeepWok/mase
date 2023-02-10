# This script tests the adder tree
import random, os, math, logging, sys

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

logger = logging.getLogger('tb_signals')

# Uncomment the following line for debugging
# logger.setLevel(logging.DEBUG)


# A source node that randomly sends out a finite number of
# data using handshake interface
class rand_source:

    def __init__(self, samples=10, num=1, maxstalls=100):
        self.num = num
        self.samples = samples
        self.maxstalls = maxstalls
        self.data = [[random.randint(0, 30) for _ in range(num)]
                     for _ in range(samples)]
        self.stallcount = 0

    def compute(self, nextready):
        tofeed = (not self.isempty()) and nextready
        if not tofeed:
            logger.debug(
                'a source cannot feed any token because of back pressure.')
            return 0, [random.randint(0, 30) for _ in range(self.num)]
        # randomly stops feeding data before reaching the max stalls
        trystall = random.randint(0, 1)
        self.stallcount += trystall
        if (not trystall) or self.stallcount > self.maxstalls:
            data = self.data[-1]
            self.data.pop()
            logger.debug(
                'a source feeds a token. Current depth = {}/{}'.format(
                    len(self.data), self.samples))
            return 1, data
        logger.debug('a source skips an iteration.')
        return 0, [random.randint(0, 30) for _ in range(self.num)]

    def isempty(self):
        return (len(self.data) == 0)


# A sink node that randomly absorbs a finite number of
# data using handshake interface
class rand_sink:

    def __init__(self, samples=10, num=1, maxstalls=100):
        self.data = []
        self.num = num
        self.samples = samples
        self.maxstalls = maxstalls
        self.stallcount = 0

    def compute(self, prevalid, datain):
        toabsorb = (not self.isfull()) and prevalid
        if not toabsorb:
            logger.debug(
                'a sink cannot absorb any token because of no valid data.')
            return 0
        # randomly stops absorbing data before reaching the max stalls
        trystall = random.randint(0, 1)
        self.stallcount += trystall
        if (not trystall) or self.stallcount > self.maxstalls:
            self.data.append(datain)
            logger.debug(
                'a sink absorbs a token. Current depth = {}/{}'.format(
                    len(self.data), self.samples))
            return 1
        logger.debug('a sink skips an iteration.')
        return 0

    def isfull(self):
        return len(self.data) == self.samples


class VerificationCase:

    def __init__(self, samples=10):
        self.in_width = 32
        self.num = 8  # 13
        self.out_width = math.ceil(math.log2(self.num)) + 32
        self.inputs = rand_source(samples=samples,
                                  num=self.num,
                                  maxstalls=2 * samples)
        self.outputs = rand_sink(samples=samples,
                                 num=self.num,
                                 maxstalls=2 * samples)
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


def checkresults(hw_out, sw_out):
    if len(hw_out) != len(sw_out):
        print("Mismatched output size: {} expected = {}".format(
            len(hw_out), len(sw_out)))
        return False
    for i in range(len(hw_out)):
        if hw_out[i] != sw_out[i]:
            print("Mismatched output value {}: {} expected = {}".format(
                i, int(hw_out[i]), sw_out[i]))
            return False
    return True


# Check if an impossible state is reached
def impossiblestate(in_ready, in_valid, out_ready, out_valid):
    # (0, X, 0, 0)
    # (0, X, 1, 0)
    # (0, X, 1, 1)
    if (not in_ready) and not ((not out_ready) and out_valid):
        return True


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
        assert not impossiblestate(
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
        done = test_case.inputs.isempty() and test_case.outputs.isfull()

    checkresults(test_case.outputs.data, test_case.ref)


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
