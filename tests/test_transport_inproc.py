"""In-proc transport: round-robin within group, fanout across groups."""
import asyncio

import pytest

from inferops.adapters.transport.inproc import InProcTransport
from inferops.core.errors import PayloadTooLarge


@pytest.mark.asyncio
async def test_inproc_roundtrip():
    t = InProcTransport()
    cons = await t.consumer("c1")
    await cons.subscribe(["topic.a"], group="g1")
    prod = await t.producer("p1")
    await prod.send("topic.a", b"hello", key=b"k")
    stream = cons.stream()
    msg = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
    assert msg.value == b"hello"
    await cons.close()
    await t.close()


@pytest.mark.asyncio
async def test_inproc_fanout_across_groups():
    t = InProcTransport()
    c1 = await t.consumer("c1")
    await c1.subscribe(["x"], group="g1")
    c2 = await t.consumer("c2")
    await c2.subscribe(["x"], group="g2")
    p = await t.producer("p")
    await p.send("x", b"v")
    m1 = await asyncio.wait_for(c1.stream().__anext__(), timeout=1.0)
    m2 = await asyncio.wait_for(c2.stream().__anext__(), timeout=1.0)
    assert m1.value == b"v" and m2.value == b"v"
    await c1.close()
    await c2.close()
    await t.close()


@pytest.mark.asyncio
async def test_inproc_rejects_oversized_payloads():
    t = InProcTransport(max_value_bytes=2)
    p = await t.producer("p")
    with pytest.raises(PayloadTooLarge):
        await p.send("x", b"too-large")
