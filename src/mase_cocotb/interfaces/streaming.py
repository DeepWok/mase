import random

import numpy as np

from cocotb.binary import BinaryValue
from cocotb.triggers import *

from mase_cocotb.driver import Driver
from mase_cocotb.monitor import Monitor


def _sign_extend(value: int, bits: int):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


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

    async def _driver_send(self, transaction) -> None:
        while True:
            await RisingEdge(self.clk)
            if type(self.data) == tuple:
                # Drive multiple data bus
                for wire, val in zip(self.data, transaction):
                    wire.value = val
            else:
                # Drive single data
                self.data.value = transaction
            if random.random() > self.valid_prob:
                self.valid.value = 0
                continue  # Try roll random valid again at next clock
            self.valid.value = 1
            await ReadOnly()
            if self.ready.value == 1:
                if type(self.data) == tuple:
                    # Drive multiple data bus
                    for t in transaction:
                        self.log.debug("Sent %s" % t)
                else:
                    self.log.debug("Sent %s" % transaction)
                break

        if self.send_queue.empty():
            await RisingEdge(self.clk)
            self.valid.value = 0


class StreamMonitor(Monitor):
    def __init__(self, clk, data, valid, ready, check=True, name=None, unsigned=False):
        super().__init__(clk, check=check, name=name)
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.check = check
        self.name = name
        self.unsigned = unsigned

    def _trigger(self):
        if "x" in self.valid.value.binstr or "x" in self.ready.value.binstr:
            return False
        return self.valid.value == 1 and self.ready.value == 1

    def _recv(self):

        def _get_sig_value(sig):

            if type(sig.value) == list:
                if self.unsigned:
                    return [x.integer for x in sig.value]
                else:
                    return [x.signed_integer for x in sig.value]

            elif type(sig.value) == BinaryValue:
                if self.unsigned:
                    return int(sig.value.integer)
                else:
                    return int(sig.value.signed_integer)

        if type(self.data) == tuple:
            # Multiple synchronised data signals
            return tuple(_get_sig_value(s) for s in self.data)
        else:
            # Single data signal
            return _get_sig_value(self.data)

    def _check(self, got, exp):

        def _check_sig(got, exp):
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
            else:
                self.log.debug(
                    "Passed | %s: \nGot \n%s, \nExpected \n%s"
                    % (
                        self.name if self.name != None else "Unnamed StreamMonitor",
                        got,
                        exp,
                    )
                )

        if self.check:
            if type(self.data) == tuple:
                assert type(got) == tuple
                assert type(exp) == tuple
                assert len(got) == len(exp), "Got & Exp Tuples are different length"
                for g, e in zip(got, exp):
                    _check_sig(g, e)
            else:
                _check_sig(got, exp)


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
                g = _sign_extend(g, self.width)
                e = _sign_extend(e, self.width)
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
                g = _sign_extend(g, self.width)
                e = _sign_extend(e, self.width)
            err = abs(g - e)
            if self.log_error:
                self.error_log.append(err)
                self.recv_log.append(got)
            if not err <= self.error_bits:
                self.log.error("Failed | Got: %20s Exp: %20s Err: %10s" % (g, e, err))
                assert fail, "Test Failed!"
                return

        else:
            g, e = got, exp
            err = np.abs(g - e)
            self.log.debug("Passed | Got: %20s Exp: %20s Err: %10s" % (g, e, err))
