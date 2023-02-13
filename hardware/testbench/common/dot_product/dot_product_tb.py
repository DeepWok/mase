# This script tests the dot product
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

    def __init__(self, samples=10, num=1, maxstalls=100, name=''):
        self.name = name
        self.num = num
        self.samples = samples
        self.maxstalls = maxstalls
        self.data = [
            [random.randint(0, 30) for _ in range(num)] for _ in range(samples)]
        self.stallcount = 0
        # Buffer the random choice
        self.random_buff = 0

    def precompute(self, nextready):
        """ A pre-compute is needed to simulate the combinational update in handshake logic"""
        tofeed = (not self.isempty()) and nextready
        if not tofeed:
            logger.debug(
                'precompute: source {} cannot feed any token because of back pressure.'
                .format(self.name))
            return 0
        # randomly stops feeding data before reaching the max stalls
        self.random_buff = random.randint(0, 1)
        self.stallcount += self.random_buff
        if (not self.random_buff) or self.stallcount > self.maxstalls:
            return 1
        logger.debug('precompute: source {} skips an iteration.'.format(
            self.name))
        return 0

    def compute(self, nextready):
        """ The actual compute computes the data as well"""
        tofeed = (not self.isempty()) and nextready
        if not tofeed:
            logger.debug(
                'source {} cannot feed any token because of back pressure.'.
                format(self.name))
            return 0, [random.randint(0, 30) for _ in range(self.num)]
        if (not self.random_buff) or self.stallcount > self.maxstalls:
            data = self.data[-1]
            self.data.pop()
            logger.debug(
                'source {} feeds a token. Current depth = {}/{}'.format(
                    self.name, len(self.data), self.samples))
            return 1, data
        logger.debug('source {} skips an iteration.'.format(self.name))
        return 0, [random.randint(0, 30) for _ in range(self.num)]

    def isempty(self):
        return (len(self.data) == 0)


# A sink node that randomly absorbs a finite number of
# data using handshake interface
class rand_sink:

    def __init__(self, samples=10, num=1, maxstalls=100, name=''):
        self.data = []
        self.name = name
        self.num = num
        self.samples = samples
        self.maxstalls = maxstalls
        self.stallcount = 0

    def compute(self, prevalid, datain):
        toabsorb = (not self.isfull()) and prevalid
        if not toabsorb:
            logger.debug(
                'a sink {} cannot absorb any token because of no valid data.'.
                format(self.name))
            return 0
        # randomly stops absorbing data before reaching the max stalls
        trystall = random.randint(0, 1)
        self.stallcount += trystall
        if (not trystall) or self.stallcount > self.maxstalls:
            self.data.append(datain)
            logger.debug(
                'sink {} absorbs a token. Current depth = {}/{}'.format(
                    self.name, len(self.data), self.samples))
            return 1
        logger.debug('sink {} skips an iteration.'.format(self.name))
        return 0

    def isfull(self):
        return len(self.data) == self.samples


class VerificationCase:

    def __init__(self, samples=10):
        self.act_width = 32
        self.w_width = 16
        self.vector_size = 2
        self.register_levels = 1
        self.act = rand_source(name='act',
                               samples=samples,
                               num=self.vector_size,
                               maxstalls=2 * samples)
        self.w = rand_source(name='w',
                             samples=samples,
                             num=self.vector_size,
                             maxstalls=2 * samples)
        self.outputs = rand_sink(samples=samples, maxstalls=2 * samples)
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
def impossiblestate(w_ready, w_valid, act_ready, act_valid, out_ready,
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
        assert not impossiblestate(
            dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
            dut.act_valid.value, dut.out_ready.value, dut.out_valid.value
        ), 'Error: invalid state (w_ready,w_valid,act_ready,act_valid,out_ready,out_valid) = ({},{},{},{},{},{})'.format(
            dut.w_ready.value, dut.w_valid.value, dut.act_ready.value,
            dut.act_valid.value, dut.out_ready.value, dut.out_valid.value)

        dut.w_valid.value = test_case.w.precompute(dut.w_ready.value)
        dut.act_valid.value = test_case.act.precompute(dut.act_ready.value)
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

        if test_case.w.isempty() and test_case.act.isempty(
        ) and test_case.outputs.isfull():
            done = True
            break
    assert done, 'Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)'

    checkresults(test_case.outputs.data, test_case.ref)


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
