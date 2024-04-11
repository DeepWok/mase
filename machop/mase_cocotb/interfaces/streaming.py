import random

import numpy as np

from cocotb.binary import BinaryValue
from cocotb.triggers import *

from mase_cocotb.driver import Driver
from mase_cocotb.monitor import Monitor
from mase_cocotb.utils import sign_extend


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
    def __init__(self, clk, data, valid, ready, check=True, name=None):
        super().__init__(clk, check=check, name=name)
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.check = check
        self.name = name

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
                self.log.error(
                    "%s: \nGot \n%s, \nExpected \n%s"
                    % (
                        self.name if self.name != None else "Unnamed StreamMonitor",
                        got,
                        exp,
                    )
                )
                assert False, "Test Failed!"


class ErrorThresholdStreamMonitor(StreamMonitor):
    def __init__(
        self,
        clk,
        data,
        valid,
        ready,
        width: int,  # Width of the number
        signed: bool,  # Signedness of number
        error_bits: int,  # Number of last bits the number can be off by
        log_error=False,  # Keep note of all errors
        check=True,
        name=None,
    ):
        super().__init__(clk, data, valid, ready, check, name)

        self.width = width
        self.signed = signed
        self.error_bits = error_bits
        self.error_log = [] if log_error else None
        self.recv_log = [] if log_error else None
        self.log_error = log_error
        self.log.setLevel("INFO")

    def _check(self, got, exp):
        fail = not self.check
        if type(got) != type(exp):
            assert fail, f"Type Mismatch got:{type(got)} vs. exp:{type(exp)}"

        # Compare Outputs
        if type(got) == list:
            g = np.array(got)
            e = np.array(exp)
            if self.signed:
                g = sign_extend(g, self.width)
                e = sign_extend(e, self.width)
            err = np.abs(g - e)
            if self.log_error:
                self.error_log.append(err)
                self.recv_log.append(got)
            max_biterr = np.full_like(err, self.error_bits)
            if not (err <= max_biterr).all():
                self.log.error("Failed | Got: %20s Exp: %20s Err: %14s" % (g, e, err))
                assert fail, "Test Failed!"
                return

        elif type(got) == int:
            g, e = got, exp
            if self.signed:
                g = sign_extend(g, self.width)
                e = sign_extend(e, self.width)
            err = abs(g - e)
            if self.log_error:
                self.error_log.append(err)
                self.recv_log.append(got)
            if not err <= self.error_bits:
                self.log.error("Failed | Got: %20s Exp: %20s Err: %10s" % (g, e, err))
                assert fail, "Test Failed!"
                return

        self.log.debug("Passed | Got: %20s Exp: %20s Err: %10s" % (g, e, err))
