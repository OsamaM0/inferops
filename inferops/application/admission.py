"""Round-robin admission across sessions for each execution resource.

Queue deadlines bound waiting, not native GPU execution. The caller retains a
permit until its operation drains, including when the caller is cancelled.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from math import isfinite

execution_key = ContextVar("execution_key", default="direct")
execution_deadline = ContextVar("execution_deadline", default=None)


class AdmissionExpired(TimeoutError):
    pass


class FairAdmission:
    def __init__(self, capacity: int, *, max_pending: int = 1024, max_queue_ms: float = 5000):
        if capacity < 1 or max_pending < 1 or max_queue_ms <= 0 or not isfinite(max_queue_ms):
            raise ValueError("admission capacities and queue deadline must be positive")
        self.capacity, self.available = capacity, capacity
        self.max_pending, self.max_queue_ms = max_pending, max_queue_ms
        self.queues = OrderedDict()
        self.pending = 0
        self.expired = 0
        self.last_key = None

    def _dispatch(self):
        while self.available and self.queues:
            if len(self.queues) > 1 and next(iter(self.queues)) == self.last_key:
                self.queues.move_to_end(self.last_key)
            key, queue = self.queues.popitem(last=False)
            future = queue.popleft()
            self.pending -= 1
            if queue:
                self.queues[key] = queue
            if not future.done():
                self.available -= 1
                self.last_key = key
                future.set_result(None)

    async def __aenter__(self):
        deadline = execution_deadline.get()
        remaining = self.max_queue_ms / 1000
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        if remaining <= 0 or self.pending >= self.max_pending:
            self.expired += 1
            raise AdmissionExpired("inference admission expired or queue is full")
        key = execution_key.get()
        future = asyncio.get_running_loop().create_future()
        self.queues.setdefault(key, deque()).append(future)
        self.pending += 1
        self._dispatch()
        try:
            await asyncio.wait_for(asyncio.shield(future), remaining)
        except BaseException as error:
            if future.done() and not future.cancelled():
                self.release()
            else:
                future.cancel()
                queue = self.queues.get(key)
                if queue is not None and future in queue:
                    queue.remove(future)
                    self.pending -= 1
                    if not queue:
                        self.queues.pop(key, None)
            if isinstance(error, TimeoutError):
                self.expired += 1
                raise AdmissionExpired("inference admission deadline exceeded") from error
            raise
        return self

    def release(self):
        self.available += 1
        self._dispatch()

    async def __aexit__(self, *args):
        self.release()

    def diagnostics(self):
        return {"capacity": self.capacity, "active": self.capacity - self.available,
                "pending": self.pending, "max_pending": self.max_pending,
                "max_queue_ms": self.max_queue_ms, "expired": self.expired}
