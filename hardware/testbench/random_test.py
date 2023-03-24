#!/usr/bin/env python3

# This script inlcudes the input and output components that have random behaviours.
# They are used for capturing all the possible dataflow computation behaviours.
import random, os, math, logging, sys


# A source node that randomly sends out a finite number of
# data using handshake interface
class RandomSource:
    def __init__(
        self,
        samples=10,
        num=1,
        max_stalls=100,
        is_data_vector=True,
        name="",
        debug=False,
    ):
        assert num > 0, "Invalid num for source {}".format(name)
        self.logger = logging.getLogger(name)
        if debug:
            logger_level = logging.DEBUG
        else:
            logger_level = logging.INFO
        self.logger.setLevel(logger_level)
        self.name = name
        self.num = num
        self.samples = samples
        self.max_stalls = max_stalls
        self.is_data_vector = is_data_vector
        if is_data_vector:
            self.data = [
                [random.randint(0, 30) for _ in range(num)] for _ in range(samples)
            ]
        else:
            self.data = [
                random.randint(0, 30) for _ in range(num) for _ in range(samples)
            ]
        self.stall_count = 0
        # Buffer the random choice
        self.random_buff = 0

    def pre_compute(self, next_ready):
        """The pre-compute simulates the combinational computation in handshake logic"""
        to_feed = (not self.is_empty()) and next_ready
        if not to_feed:
            self.logger.debug(
                "pre_compute: source {} cannot feed any token because of back pressure.".format(
                    self.name
                )
            )
            return 0
        # randomly stops feeding data before reaching the max stalls
        self.random_buff = random.randint(0, 1)
        self.stall_count += self.random_buff
        if (not self.random_buff) or self.stall_count > self.max_stalls:
            return 1
        self.logger.debug(
            "pre_compute: source {} skips an iteration.".format(self.name)
        )
        return 0

    def compute(self, next_ready):
        """The compute simulates the synchronous computation for data"""
        to_feed = (not self.is_empty()) and next_ready
        if self.is_data_vector:
            dummy_out = [random.randint(0, 30) for _ in range(self.num)]
        else:
            dummy_out = random.randint(0, 30)
        if not to_feed:
            self.logger.debug(
                "source {} cannot feed any token because of back pressure.".format(
                    self.name
                )
            )
            return 0, dummy_out
        if (not self.random_buff) or self.stall_count > self.max_stalls:
            data = self.data[-1]
            self.data.pop()
            self.logger.debug(
                "source {} feeds a token. Current depth = {}/{}".format(
                    self.name, len(self.data), self.samples
                )
            )
            return 1, data
        self.logger.debug("source {} skips an iteration.".format(self.name))
        return 0, dummy_out

    def is_empty(self):
        return len(self.data) == 0


# A sink node that randomly absorbs a finite number of
# data using handshake interface
class RandomSink:
    def __init__(self, samples=10, num=1, max_stalls=100, name="", debug=False):
        assert num > 0, "Invalid num for sink {}".format(name)
        self.logger = logging.getLogger(name)
        if debug:
            logger_level = logging.DEBUG
        else:
            logger_level = logging.INFO
        self.logger.setLevel(logger_level)
        self.data = []
        self.name = name
        self.num = num
        self.samples = samples
        self.max_stalls = max_stalls
        self.stall_count = 0

    def compute(self, prevalid, datain):
        to_absorb = (not self.is_full()) and prevalid
        if not to_absorb:
            self.logger.debug(
                "a sink {} cannot absorb any token because of no valid data.".format(
                    self.name
                )
            )
            return 0
        # randomly stops absorbing data before reaching the max stalls
        trystall = random.randint(0, 1)
        self.stall_count += trystall
        if (not trystall) or self.stall_count > self.max_stalls:
            self.data.append(datain)
            self.logger.debug(
                "sink {} absorbs a token. Current depth = {}/{}".format(
                    self.name, len(self.data), self.samples
                )
            )
            return 1
        self.logger.debug("sink {} skips an iteration.".format(self.name))
        return 0

    def is_full(self):
        return len(self.data) == self.samples


def check_results(hw_out, sw_out):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
    for i in range(len(hw_out)):
        assert (
            hw_out[i] == sw_out[i]
        ), "Mismatched output value {}: {} expected = {}".format(
            i, [int(t) for t in hw_out[i]], [int(t) for t in sw_out[i]]
        )
    return True
