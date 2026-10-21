"""Owner-local control behind an application port."""
from typing import Protocol


class WorkerControl(Protocol):
    async def status(self) -> dict: ...
    async def close(self) -> None: ...
