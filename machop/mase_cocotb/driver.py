from queue import Queue

import cocotb
from cocotb.decorators import coroutine
from cocotb.triggers import *


class Driver:
    """Simplified version of cocotb_bus.drivers.Driver"""

    def __init__(self):
        self._pending = Event(name="Driver._pending")
        self.send_queue = Queue()

        if not hasattr(self, "log"):
            self.log = SimLog("cocotb.driver.%s" % (type(self).__qualname__))

        # Create an independent coroutine which can send stuff
        self._thread = cocotb.start_soon(self._send_thread())

    def kill(self):
        if self._thread:
            self._thread.kill()
            self._thread = None

    def append(self, transaction) -> None:
        self.send_queue.put(transaction)
        self._pending.set()

    async def _send_thread(self):
        while True:
            # Sleep until we have something to send
            while self.send_queue.empty():
                self._pending.clear()
                await self._pending.wait()

            # Send in all the queued packets,
            # only synchronize on the first send
            while not self.send_queue.empty():
                transaction = self.send_queue.get()
                await self.send(transaction)

    def clear(self):
        self.send_queue = Queue()

    @coroutine
    async def send(self, transaction) -> None:
        """Blocking send call (hence must be "awaited" rather than called).

        Sends the transaction over the bus.

        Args:
            transaction: The transaction to be sent.
            sync: Synchronize the transfer by waiting for a rising edge.
            **kwargs: Additional arguments used in child class'
                :any:`_driver_send` method.
        """
        await self._driver_send(transaction)

    async def _driver_send(self, transaction: Any) -> None:
        raise NotImplementedError(
            "Sub-classes of Driver should define a " "_driver_send coroutine"
        )
