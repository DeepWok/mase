from queue import Queue

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure


class Monitor:
    """Simplified version of cocotb_bus.monitors.Monitor"""

    def __init__(self, clk, check=True, name=None):
        self.clk = clk
        self.recv_queue = Queue()
        self.exp_queue = Queue()
        self.check = check
        self.name = name
        self.in_flight = False

        if not hasattr(self, "log"):
            self.log = SimLog(
                "cocotb.monitor.%s" % (type(self).__qualname__)
                if self.name == None
                else self.name
            )

        self._thread = cocotb.scheduler.add(self._recv_thread())

    def kill(self):
        if self._thread:
            self._thread.kill()
            self._thread = None

    def expect(self, transaction):
        self.exp_queue.put(transaction)

    async def _recv_thread(self):
        while True:
            await RisingEdge(self.clk)
            if self._trigger():
                tr = self._recv()
                self.log.debug(f"Observed output beat {tr}")
                self.recv_queue.put(tr)

                if self.exp_queue.empty():
                    assert False, (
                        "Got %s but we did not expect anything." % self.recv_queue.get()
                    )

                self._check(self.recv_queue.get(), self.exp_queue.get())

                # * If the monitor is in-flight (expectation queue has been populated)
                # * and the expectation queue is now empty (after running the check),
                # * the test is finished
                if (
                    self.in_flight == True
                    and self.recv_queue.empty()
                    and self.exp_queue.empty()
                ):
                    self.in_flight = False
                    self.log.info(f"Monitor has been drained.")

    def _trigger(self):
        raise NotImplementedError()

    def _recv(self):
        raise NotImplementedError()

    def _check(self, got, exp):
        raise NotImplementedError()

    def clear(self):
        self.send_queue = Queue()

    def load_monitor(self, tensor):
        for beat in tensor:
            self.log.debug(f"Expecting output beat {beat}")
            self.expect(beat)
