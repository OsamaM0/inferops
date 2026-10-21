"""Host-local process exclusion for a static shard's durable spool.

Leases fence SQL effects; this lock additionally prevents a returning process's
uninterruptible filesystem write from racing a replacement owner. Cross-host
automatic shard reassignment is deliberately unsupported.
"""
from filelock import FileLock, Timeout

from inferops.adapters.outbound.media.spool import CrashSafeSpool


class OwnedSpool(CrashSafeSpool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_lock = None

    def _start(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self.process_lock is None:
            lock = FileLock(self.path / ".process-owner.lock", thread_local=False)
            try:
                lock.acquire(timeout=0)
            except Timeout as error:
                raise RuntimeError("shard spool still has a live process owner; stop that process before recovery") from error
            self.process_lock = lock
        try:
            super()._start()
        except BaseException:
            self.process_lock.release()
            self.process_lock = None
            raise

    async def close(self):
        if self.process_lock is not None:
            await self._run(self.process_lock.release)
            self.process_lock = None
