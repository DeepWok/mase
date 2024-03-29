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

    def postprocess_tensor(self, tensor, config):
        tensor = [item * (1.0 / 2.0) ** config["frac_width"] for item in tensor]
        return tensor

    def _check(self, got, exp):
        if self.check:
            print("_check: ", self.postprocess_tensor(got, {"frac_width": 3}))
            if not np.equal(got, exp).all():
                raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))


class StreamMonitorRange(Monitor):
    def __init__(
        self, clk, data, valid, ready, out_width=8, out_frac_width=3, check=True
    ):
        super().__init__(clk)
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.check = check
        self.out_width = out_width
        self.out_frac_width = out_frac_width

    def _trigger(self):
        return self.valid.value == 1 and self.ready.value == 1

    def _recv(self):
        if type(self.data.value) == list:
            return [int(x) for x in self.data.value]
        elif type(self.data.value) == BinaryValue:
            return int(self.data.value)

    def postprocess_tensor(self, tensor, frac_width):
        tensor = [item * (1.0 / 2.0) ** frac_width for item in tensor]
        return tensor

    def _check(self, got, exp):
        if self.check:
            print("_check: ", self.postprocess_tensor(got, 3))
            if len(got) != len(exp):
                raise TestFailure(
                    "\nGot \n%s, does not match dimension of \nExpected \n%s"
                    % (got, exp)
                )

            for exp_val, got_val in zip(exp, got):
                if exp_val >= 128:
                    continue
                    # exp_val = exp_val * (1.0/2.0) ** self.out_frac_width
                    # exp_val = exp_val - (2 ** (self.out_width-self.out_frac_width))
                else:
                    exp_val = exp_val * (1.0 / 2.0) ** self.out_frac_width

                if got_val >= 128:
                    continue
                    # got_val = got_val * (1.0/2.0) ** self.out_frac_width
                    # got_val = got_val - (2 ** (self.out_width-self.out_frac_width))
                else:
                    got_val = got_val * (1.0 / 2.0) ** self.out_frac_width

                error = abs(exp_val - got_val)
                print(f"{got_val},{exp_val},{error}")
                if error > 2.0:
                    raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))
            # if not np.equal(got, exp).all():
            #     raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))
