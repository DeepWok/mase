from queue import Queue

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure


class Monitor:
    """Simplified version of cocotb_bus.monitors.Monitor"""

    def __init__(self, clk, check=True):
        self.clk = clk
        self.recv_queue = Queue()
        self.exp_queue = Queue()
        self.check = check

        if not hasattr(self, "log"):
            self.log = SimLog("cocotb.monitor.%s" % (type(self).__qualname__))

        self._thread = cocotb.start_soon(self._recv_thread())

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
                self.log.info(f"Observed output beat {tr}")
                self.recv_queue.put(tr)

                if self.exp_queue.empty():
                    raise TestFailure(
                        "\nGot \n%s,\nbut we did not expect anything."
                        % self.recv_queue.get()
                    )

                self._check(self.recv_queue.get(), self.exp_queue.get())

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
            self.log.info(f"Expecting output beat {beat}")
            self.expect(beat)
