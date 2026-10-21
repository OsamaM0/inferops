"""Capacity recommendations bounded by matching observed measurements."""

from __future__ import annotations

import math


def assess_capacity(
    reports,
    hardware,
    *,
    pipeline,
    runtime,
    model_artifacts,
    width,
    height,
    fps,
    evidence,
    workload_fingerprint=None,
    evidence_policy=None,
    software_fingerprint=None,
    service_profile=None,
    evidence_fps=None,
):
    matches = []
    current_uuids = {g["uuid"] for g in hardware.get("gpus", [])}
    for report in reports:
        if not report.get("completed"):
            continue
        if service_profile is not None and report.get("service_profile") != service_profile:
            continue
        if software_fingerprint is not None and report.get("software_fingerprint") != software_fingerprint:
            continue
        if workload_fingerprint is not None and report.get("workload_fingerprint") != workload_fingerprint:
            continue
        if evidence_policy is not None and report.get("evidence_policy") != evidence_policy:
            continue
        if report.get("pipeline") != pipeline or report.get("runtime") != runtime:
            continue
        if report.get("model_artifacts") != model_artifacts:
            continue
        host = report.get("hardware", {})
        if hardware.get("data_filesystem") and host.get("data_filesystem") != hardware["data_filesystem"]:
            continue
        measured_ram, current_ram = host.get("ram_bytes"), hardware.get("ram_bytes")
        # Linux may report a few pages differently after boot/device changes.
        # Preserve the installed-memory class without rejecting an 8 KiB drift.
        same_ram = measured_ram == current_ram or (
            isinstance(measured_ram, (int, float)) and isinstance(current_ram, (int, float))
            and measured_ram > 0 and current_ram > 0
            and abs(measured_ram - current_ram) / max(measured_ram, current_ram) <= 0.01
        )
        if (host.get("cpu_model") != hardware.get("cpu_model")
                or host.get("cpu_count") != hardware.get("cpu_count") or not same_ram):
            continue
        if runtime != "cpu" and (
            not current_uuids or {g["uuid"] for g in host.get("gpus", [])} != current_uuids
        ):
            continue
        if bool(report.get("evidence_enabled")) != evidence or not report.get("preview_enabled"):
            continue
        for case in report.get("cases", []):
            # No pixel/FPS extrapolation: model inputs and codec costs are nonlinear.
            if (
                case.get("width") == width
                and case.get("height") == height
                and case.get("inference_fps") == fps
                and max(case.get("inference_fps", 0), case.get("evidence_fps") or case.get("inference_fps", 0))
                == max(fps, evidence_fps or fps)
            ):
                matches.append(
                    {
                        **case,
                        "report_id": report.get("report_id"),
                        "created_at": report.get("created_at"),
                        "source_kind": report.get("source_kind"),
                        "limitations": report.get("limitations", []),
                    }
                )
    passed = [c for c in matches if c.get("passed")]
    highest = max((c["cameras"] for c in passed), default=None)
    return {
        "status": "measured_local_video"
        if highest is not None
        else "measured_below_target"
        if matches
        else "unmeasured",
        "tested_passing_cameras": highest,
        "suggested_cameras_with_20_percent_headroom": math.floor(highest * 0.8) if highest and highest > 1 else None,
        "first_tested_failure": min((c["cameras"] for c in matches if not c.get("passed")), default=None),
        "maximum_cameras": None,
        "explanation": "Short measured sweeps are not a certified maximum. Validate RTSP, evidence bursts and a long soak on your workload."
        if matches
        else "No matching hardware/model/runtime/resolution/FPS/evidence profile. Run a benchmark; VRAM alone cannot determine camera capacity.",
        "cases": matches,
    }
