"""Fault-check an isolated worker deployment with real controls, models and storage.

Docker fault targets must start with inferops-plan-test to prevent accidentally
stopping an ordinary deployment. The specified cameras must already be running.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import httpx


def verify(args):
    for name in [args.api_container, *args.worker_containers, args.postgres_container, args.minio_container]:
        if name and not name.startswith("inferops-plan-test"):
            raise ValueError("fault injection is restricted to isolated inferops-plan-test containers")
    root = args.output / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root.mkdir(parents=True)
    report = {"at": time.time(), "url": args.url, "camera_ids": args.camera_ids, "checks": [], "passed": False}
    client = httpx.Client(base_url=args.url, timeout=15)
    paused = set()

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2))

    def cameras():
        response = client.get("/v1/cameras")
        response.raise_for_status()
        return {row["camera_id"]: row for row in response.json()["cameras"] if row["camera_id"] in args.camera_ids}

    def wait_for(predicate, *, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                rows = cameras()
                if len(rows) == len(args.camera_ids) and predicate(rows):
                    return rows
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise AssertionError("deployment did not reach the required state before its recovery deadline")

    def docker(*parts):
        subprocess.run(["docker", *parts], check=True, capture_output=True, timeout=30)

    def healthy(rows):
        return all(row.get("active") and row.get("processed", 0) > 0 and row.get("status") == "running"
                   and row.get("last_result_at", 0) > time.time() - 5 for row in rows.values())

    def check(name, began, **values):
        report["checks"].append({"name": name, "seconds": time.monotonic() - began, "passed": True, **values})
        save()

    try:
        before = wait_for(healthy)
        began = time.monotonic()
        time.sleep(args.steady_seconds)
        after = wait_for(healthy)
        assert all(after[key]["processed"] > before[key]["processed"] for key in args.camera_ids)
        for key in args.camera_ids:
            response = client.get(f"/v1/cameras/{key}/preview.jpg")
            assert response.status_code == 200 and response.headers["content-type"].startswith("image/jpeg")
            assert client.get(f"/v1/cameras/{key}/results").json()
        check("healthy_streams_previews_and_history", began, cameras=after)

        before = after
        began = time.monotonic()
        docker("restart", "--time", "10", args.api_container)
        after = wait_for(lambda rows: healthy(rows) and all(rows[key]["processed"] > before[key]["processed"] for key in args.camera_ids))
        assert all(after[key]["session_id"] == before[key]["session_id"] for key in args.camera_ids)
        check("API_restart_preserves_worker_sessions", began)

        before = after
        began = time.monotonic()
        # A manual Docker kill suppresses restart, and Linux protects namespace
        # PID 1 from signals sent inside that namespace. A minimal host-PID helper
        # kills only the inspected test container process, without host mounts,
        # network access or capabilities other than KILL.
        target = json.loads(subprocess.check_output(["docker", "inspect", args.worker_containers[0]]))[0]
        pid = target["State"]["Pid"]
        if not target["State"]["Running"] or pid <= 1:
            raise RuntimeError("test worker has no running process to crash")
        docker("run", "--rm", "--pid=host", "--network=none", "--read-only",
               "--cap-drop=ALL", "--cap-add=KILL", "--user=0", "--entrypoint=python",
               target["Image"], "-c", f"import os,signal; os.kill({pid}, signal.SIGKILL)")
        after = wait_for(lambda rows: healthy(rows) and rows[args.camera_ids[0]]["session_id"] != before[args.camera_ids[0]]["session_id"], timeout=60)
        for key in args.camera_ids[1:]:
            assert after[key]["session_id"] == before[key]["session_id"]
            assert after[key]["processed"] > before[key]["processed"]
        check("worker_kill_recovers_own_shard_only", began)

        for container, seconds, name in [(args.minio_container, 30, "MinIO_30_second_outage"),
                                          (args.postgres_container, 25, "PostgreSQL_lease_expiry")]:
            if not container:
                continue
            began = time.monotonic()
            docker("pause", container)
            paused.add(container)
            time.sleep(seconds)
            docker("unpause", container)
            paused.remove(container)
            after = wait_for(healthy, timeout=150)
            check(name, began, cameras=after)
        for key in args.camera_ids:
            response = client.post(f"/v1/cameras/{key}/stop")
            response.raise_for_status()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            records = client.get("/v1/evidence/records", params={"limit": 1000}).json()
            if records and all(row["state"] != "Pending" for row in records):
                break
            time.sleep(1)
        assert records and all(row["state"] in {"Saved", "Incomplete", "Failed"} for row in records)
        verified = 0
        for row in records:
            if row["state"] != "Saved":
                assert row.get("errors") or row.get("missing_intervals")
                continue
            for kind, artifact in row.get("artifacts", {}).items():
                response = client.get(f"/v1/evidence/records/{row['id']}/assets/{kind}")
                response.raise_for_status()
                assert hashlib.sha256(response.content).hexdigest() == artifact["sha256"]
                if kind == "clip":
                    import av
                    with av.open(io.BytesIO(response.content)) as media:
                        assert next(media.decode(video=0), None) is not None
            verified += 1
        report.update(passed=True, evidence_records=len(records), checksum_playback_verified=verified,
                      evidence_states={state: sum(row["state"] == state for row in records)
                                       for state in ["Saved", "Incomplete", "Failed"]})
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for container in paused:
            docker("unpause", container)
        save()
        client.close()
        print(root / "report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18082")
    parser.add_argument("--camera-ids", type=lambda value: value.split(","), required=True)
    parser.add_argument("--api-container", required=True)
    parser.add_argument("--worker-containers", type=lambda value: value.split(","), required=True)
    parser.add_argument("--postgres-container")
    parser.add_argument("--minio-container")
    parser.add_argument("--steady-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path(".run/worker-acceptance"))
    verify(parser.parse_args())
