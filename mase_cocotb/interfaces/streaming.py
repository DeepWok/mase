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
            breakpoint()
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
