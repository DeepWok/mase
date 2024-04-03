import random

import numpy as np

from cocotb.binary import BinaryValue
from cocotb.result import TestFailure
from cocotb.triggers import *

from mase_cocotb.driver import Driver
from mase_cocotb.monitor import Monitor


class StreamDriver(Driver):
    def __init__(self, clk, data, valid, ready) -> None:
        super().__init__()
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.valid_prob = 1.0

    def set_valid_prob(self, prob):
        assert prob >= 0.0 and prob <= 1.0
        self.valid_prob = prob

    async def _driver_send(self, data) -> None:
        while True:
            await RisingEdge(self.clk)
            self.data.value = data
            if random.random() > self.valid_prob:
                self.valid.value = 0
                continue  # Try roll random valid again at next clock
            self.valid.value = 1
            await ReadOnly()
            if self.ready.value == 1:
                self.log.debug("Sent %s" % data)
                break

        if self.send_queue.empty():
            await RisingEdge(self.clk)
            self.valid.value = 0


class StreamMonitor(Monitor):
    def __init__(self, clk, data, valid, ready, check=True):
        super().__init__(clk)
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.check = check

    def _trigger(self):
        return self.valid.value == 1 and self.ready.value == 1

    def _recv(self):
        if type(self.data.value) == list:
            return [int(x) for x in self.data.value]
        elif type(self.data.value) == BinaryValue:
            return int(self.data.value)

    def _check(self, got, exp):
        if self.check:
            if not np.equal(got, exp).all():
                raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))


class StreamMonitorFloat(StreamMonitor):
    def __init__(self, clk, data, valid, ready, data_width, frac_width, check=True):
        super().__init__(clk, data, valid, ready, check)
        self.data_width = data_width
        self.frac_width = frac_width

    def _check(self, got, exp):
        if self.check:
            float_got = [x * 2**-self.frac_width for x in got]
            float_exp = [x * 2**-self.frac_width for x in exp]
            if not np.isclose(float_got, float_exp, atol=2**-self.frac_width).all():
                # raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))
                raise TestFailure(
                    f"\nGot int \n{got}, \nExpected int \n{exp} \nGot float \n{float_got}, \nExpected float \n{float_exp}"
                )
