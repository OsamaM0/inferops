"""Read server hardware, with unknown values when GPU visibility is unavailable."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


def filesystem_type(path):
    try:
        target = Path(path).resolve()
        matches = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(left.split()[4].replace("\\040", " "))
            if target.is_relative_to(mount):
                matches.append((len(str(mount)), right.split()[0]))
        return max(matches)[1] if matches else None
    except (OSError, ValueError, IndexError):
        return None


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            name, uuid, total, used, free, utilization, driver = [s.strip() for s in line.split(",")]
            gpus.append(
                dict(
                    name=name,
                    uuid=uuid,
                    total_mib=float(total),
                    used_mib=float(used),
                    free_mib=float(free),
                    utilization_percent=float(utilization),
                    driver=driver,
                )
            )
        return {"available": bool(gpus), "gpus": gpus}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {
            "available": False,
            "gpus": [],
            "reason": "NVIDIA telemetry unavailable to this server process",
        }


def hardware_snapshot(data_path="."):
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    except OSError:
        cpu = platform.processor()
    disk = shutil.disk_usage(data_path)
    return {
        "at": time.time(),
        "hostname": platform.node(),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "ram_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "data_filesystem": filesystem_type(data_path),
        **gpu_snapshot(),
    }
