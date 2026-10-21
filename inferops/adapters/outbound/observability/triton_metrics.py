"""Inference GPU telemetry when the API container does not expose NVML."""

from __future__ import annotations

import httpx
from prometheus_client.parser import text_string_to_metric_families


def parse_gpu_metrics(body):
    devices = {}
    names = {
        "nv_gpu_memory_total_bytes": "total",
        "nv_gpu_memory_used_bytes": "used",
        "nv_gpu_utilization": "utilization",
    }
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name not in names or not sample.labels.get("gpu_uuid"):
                continue
            devices.setdefault(sample.labels["gpu_uuid"], {})[names[sample.name]] = sample.value
    return [
        {
            "uuid": uuid,
            "name": f"Triton GPU · {uuid[-8:]}",
            "total_mib": row["total"] / 1024**2,
            "used_mib": row["used"] / 1024**2,
            "free_mib": (row["total"] - row["used"]) / 1024**2,
            "utilization_percent": round(row.get("utilization", 0) * 100, 1),
            "driver": None,
            "telemetry_source": "triton",
        }
        for uuid, row in devices.items()
        if "total" in row and "used" in row
    ]


async def inference_gpus(url):
    try:
        async with httpx.AsyncClient(timeout=1) as client:
            response = await client.get(url.rstrip("/") + "/metrics")
            response.raise_for_status()
        return parse_gpu_metrics(response.text)
    except (httpx.HTTPError, ValueError):
        return []
