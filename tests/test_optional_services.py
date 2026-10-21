import asyncio

import pytest

from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
from inferops.adapters.outbound.observability.triton_metrics import parse_gpu_metrics


async def test_local_incident_is_visible_while_optional_broker_delivery_remains_retryable():
    class FailedBroker:
        async def producer(self, name):
            raise OSError("broker down")

    hub = DeliveryHub()
    queue = asyncio.Queue()
    hub.subscribe("alerts.camera", queue)
    delivery = ServerDelivery(hub, transport=FailedBroker())
    alert = {"id": "incident", "payload": {"stream_name": "camera"}}
    with pytest.raises(OSError, match="broker down"):
        await delivery.publish("incident.accepted", "session", {"alert": alert}, "event")
    assert queue.get_nowait() == alert
    await delivery.close()


def test_remote_gpu_memory_units_and_devices_stay_separate():
    rows = parse_gpu_metrics("""nv_gpu_memory_total_bytes{gpu_uuid="GPU-a"} 8589934592
nv_gpu_memory_used_bytes{gpu_uuid="GPU-a"} 2147483648
nv_gpu_utilization{gpu_uuid="GPU-a"} 0.6
nv_gpu_memory_total_bytes{gpu_uuid="GPU-b"} 17179869184
nv_gpu_memory_used_bytes{gpu_uuid="GPU-b"} 1073741824
""")
    assert len(rows) == 2
    assert rows[0]["total_mib"] == 8192 and rows[0]["used_mib"] == 2048
    assert rows[0]["utilization_percent"] == 60
    assert rows[1]["total_mib"] == 16384 and rows[1]["used_mib"] == 1024
