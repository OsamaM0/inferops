import asyncio
import time

import pytest

from inferops.application.admission import AdmissionExpired, FairAdmission, execution_deadline, execution_key


async def test_a_burst_from_one_session_does_not_starve_another():
    gate = FairAdmission(1)
    order = []
    await gate.__aenter__()

    async def request(key):
        token = execution_key.set(key)
        try:
            async with gate:
                order.append(key)
        finally:
            execution_key.reset(token)

    tasks = [asyncio.create_task(request(key)) for key in ["fast"] * 5 + ["slow"] * 2]
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(*tasks)
    assert order[:4] == ["fast", "slow", "fast", "slow"]
    assert gate.available == 1 and gate.pending == 0


async def test_expiry_and_waiter_cancellation_leave_no_capacity_leak():
    gate = FairAdmission(1, max_queue_ms=10)
    await gate.__aenter__()
    with pytest.raises(AdmissionExpired):
        await gate.__aenter__()
    task = asyncio.create_task(gate.__aenter__())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    gate.release()
    assert gate.available == 1 and gate.pending == 0
    token = execution_deadline.set(time.monotonic() - 1)
    try:
        with pytest.raises(AdmissionExpired):
            await gate.__aenter__()
    finally:
        execution_deadline.reset(token)
