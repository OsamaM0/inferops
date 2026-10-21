"""Real models, routed RTSP, files, evidence and recovery against local services.

Requires a running MediaMTX Compose service, Triton for --runtime triton,
FFmpeg, Docker and playwright-cli. Uses an isolated API/database and disposable
origin container; leaves the existing router/model server and user API alone.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import time
import uuid
from collections import Counter
from contextlib import suppress
from pathlib import Path

import av
import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


async def command(*args):
    process = await asyncio.create_subprocess_exec(*map(str, args), cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed: {(stderr or stdout).decode()[-1500:]}")
    return stdout.decode()


async def wait_for(check, *, timeout=45, label):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            last = await check()
            if last:
                return last
        except (httpx.HTTPError, KeyError):
            pass
        await asyncio.sleep(.25)
    raise AssertionError(f"Timed out: {label}; last={last}")


async def verify(args):
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    root = args.output.resolve() / stamp
    root.mkdir(parents=True)
    camera_id = "verify-real-" + stamp
    origin = "inferops-origin-" + uuid.uuid4().hex[:8]
    report = {"passed": False, "runtime": args.runtime, "camera_id": camera_id,
              "input": str(args.video.resolve()), "checks": {}, "started_at": time.time()}
    api = publisher = None
    log_files = []
    browser_session = "real-" + uuid.uuid4().hex[:8]

    def progress(message):
        print(message, flush=True)

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=90,
                                 trust_env=False) as client, httpx.AsyncClient(
        base_url=args.router, auth=(args.router_user, args.router_password), timeout=8, trust_env=False
    ) as media:
        async def request(method, path, **kwargs):
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        async def camera():
            return next(row for row in (await request("GET", "/v1/cameras"))["cameras"]
                        if row["camera_id"] == camera_id)

        async def processed_after(count):
            row = await camera()
            assert row.get("status") != "error", row
            return row if row.get("processed", 0) > count else None

        async def start_api():
            nonlocal api
            env = dict(os.environ)
            env.update(INFEROPS_ROLE="combined", INFEROPS_DATABASE_URL=args.database_url,
                INFEROPS_S3_BUCKET="inferops-verification", INFEROPS_MEDIA_ROUTER="mediamtx",
                INFEROPS_MEDIAMTX_API_URL=args.router, INFEROPS_MEDIAMTX_RTSP_URL=args.rtsp,
                INFEROPS_MEDIAMTX_API_USER=args.router_user, INFEROPS_MEDIAMTX_API_PASSWORD=args.router_password,
                INFEROPS_OPENCV_THREADS="1")
            for key in ("INFEROPS_WORKER_MANIFEST", "INFEROPS_WORKER_ID"):
                env.pop(key, None)
            log = (root / "api.log").open("a")
            log_files.append(log)
            api = subprocess.Popen([str(ROOT / ".venv/bin/python"), "scripts/serve_local.py",
                "--runtime", args.runtime, "--port", str(args.port), "--data", str(root / "data")],
                cwd=ROOT, env=env, stdout=log, stderr=log)

            async def ready():
                if api.poll() is not None:
                    raise AssertionError("verification API exited; inspect api.log")
                return (await client.get("/readyz")).status_code == 200

            await wait_for(ready, label="API startup")

        async def stop_api():
            if api is not None and api.poll() is None:
                api.terminate()
                await asyncio.wait_for(asyncio.to_thread(api.wait), 50)

        def start_publisher():
            nonlocal publisher
            log = (root / "publisher.log").open("a")
            log_files.append(log)
            publisher = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-re", "-stream_loop", "-1", "-i", str(args.video.resolve()), "-an", "-r", "15",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline",
                "-pix_fmt", "yuv420p", "-g", "15", "-bf", "0", "-threads", "2", "-f", "rtsp",
                "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{args.origin_port}/camera"],
                cwd=ROOT, stdout=log, stderr=log)

        async def stop_publisher():
            if publisher is not None and publisher.poll() is None:
                publisher.terminate()
                await asyncio.wait_for(asyncio.to_thread(publisher.wait), 10)

        try:
            response = await media.get("/v3/paths/list")
            response.raise_for_status()
            network = json.loads(await command("docker", "inspect", args.router_container,
                "--format", "{{json .NetworkSettings.Networks}}"))
            await command("docker", "run", "-d", "--name", origin, "--network", next(iter(network)),
                "-p", f"127.0.0.1:{args.origin_port}:8554", "-e", "MTX_RTSPTRANSPORTS=tcp",
                "-e", "MTX_WEBRTC=no", "-e", "MTX_HLS=no", "-e", "MTX_MOQ=no", "-e", "MTX_SRT=no",
                "-e", "MTX_RTMP=no", "bluenviron/mediamtx:1.21.2")
            await asyncio.sleep(.5)
            start_publisher()
            await start_api()
            before = (await client.get("http://127.0.0.1:8000/v2/models/stats")).json() if args.runtime == "triton" else None
            spec = {"camera_id": camera_id, "name": "Real driver verification", "kind": "rtsp",
                "uri": f"rtsp://{origin}:8554/camera", "pipeline": "driver_behaviour", "inference_fps": 2,
                "evidence_fps": 10, "max_width": 640, "max_height": 480, "preview_fps": 2,
                "preview_enabled": True, "autostart": True, "timeout_s": 3}
            await request("POST", "/v1/cameras", json=spec)
            await request("POST", f"/v1/cameras/{camera_id}/start")
            initial = await wait_for(lambda: processed_after(35), timeout=75, label="real inference and capture")
            session_id = initial["session_id"]
            rows = await request("GET", f"/v1/cameras/{camera_id}/results?limit=100")
            detections = [det for row in rows for output in row["results"]
                          for result in output.get("results", []) for det in result.get("detections", [])]
            assert detections, "real detector must produce foreground detections"
            assert all(row["camera_id"] == camera_id and row["session_id"] == session_id for row in rows)
            jpeg = await client.get(f"/v1/cameras/{camera_id}/preview.jpg")
            jpeg.raise_for_status()
            assert cv2.imdecode(np.frombuffer(jpeg.content, np.uint8), cv2.IMREAD_COLOR) is not None
            (root / "annotated.jpg").write_bytes(jpeg.content)
            status = await request("GET", "/v1/system/status")
            assert status["services"]["mediamtx"]["status"] == 200
            loaded = status["runtime"]["models"]
            assert loaded and not status["runtime"]["models_downgraded"], loaded
            assert all(row["device"] == ("remote" if args.runtime == "triton" else "gpu" if args.runtime == "cuda" else "cpu")
                       for row in loaded), loaded
            report["runtime_models"], report["initial"] = loaded, initial
            report["checks"]["rtsp_real_inference_annotations"] = {"processed": initial["processed"],
                "foreground_detections": len(detections), "source_size": [initial["source_width"], initial["source_height"]],
                "processing_size": [initial["processed_width"], initial["processed_height"]], "target_fps": 2,
                "observed_session_fps": initial["fps"]}
            progress("PASS real RTSP → models → detections → annotated JPEG")

            opened = await command("playwright-cli", f"-s={browser_session}", "open", str(client.base_url).rstrip("/") + "/ui/")
            (root / "browser-open.log").write_text(opened)
            # The CLI launches browser code in a separate process; supply concrete identifiers as a wrapper.
            code = (ROOT / "tests/scenarios/verify_media_router.browser.js").read_text()
            code = code.replace("__CAMERA__", json.dumps(camera_id)).replace("__OUTPUT__", json.dumps(str(root)))
            browser_file = root / "browser.js"
            browser_file.write_text(code)
            browser_result = await command("playwright-cli", f"-s={browser_session}", "run-code", f"--filename={browser_file}")
            (root / "browser.log").write_text(browser_result)
            assert '"passed":true' in browser_result, browser_result
            async def no_idle_muxer():
                muxers = (await media.get("/v3/hls/muxers/list")).json()["items"]
                return not any(row.get("path") == f"inferops/combined/{camera_id}" for row in muxers)
            closed_at = time.monotonic()
            await wait_for(no_idle_muxer, timeout=12, label="HLS idle remuxing release")
            report["checks"]["hls_remux_release_seconds"] = round(time.monotonic() - closed_at, 2)
            async def only_inference_remains():
                paths = (await media.get("/v3/paths/list")).json()["items"]
                routed = next(row for row in paths if row["name"].endswith("/" + camera_id))
                return routed if len(routed["readers"]) == 1 else None
            # Upstream HLS sessions expire after 30s inactivity with a 10s
            # check interval, separately from the shorter remuxer lifetime.
            routed = await wait_for(only_inference_remains, timeout=45, label="HLS idle session release")
            report["checks"]["browser_playback_and_release"] = True
            report["checks"]["router_readers_after_player_close"] = len(routed["readers"])
            report["checks"]["player_release_seconds"] = round(time.monotonic() - closed_at, 2)
            progress("PASS WebRTC/HLS moving video, annotated wall, player cleanup")

            count = (await camera())["processed"]
            await stop_publisher()
            async def disconnected():
                row = await camera()
                return row if row.get("connection") == "reconnecting" and row["active"] else None
            await wait_for(disconnected, timeout=25, label="camera disconnect detection")
            start_publisher()
            recovery_started = time.monotonic()
            resumed = await wait_for(lambda: processed_after(count + 5), timeout=45, label="source reconnect")
            assert resumed["session_id"] == session_id and resumed["reconnects"] >= 1
            report["checks"]["source_reconnect_seconds"] = round(time.monotonic() - recovery_started, 2)
            progress("PASS camera cut and reconnect with the same camera/session")

            count = resumed["processed"]
            await command("docker", "restart", args.router_container)
            recovery_started = time.monotonic()
            resumed = await wait_for(lambda: processed_after(count + 5), timeout=50, label="router restart recovery")
            assert resumed["session_id"] == session_id
            report["checks"]["router_restart_processing_recovery_seconds"] = round(time.monotonic() - recovery_started, 2)
            progress("PASS MediaMTX restart and real inference recovery")

            await request("POST", f"/v1/cameras/{camera_id}/stop")
            await request("PUT", f"/v1/cameras/{camera_id}", json={"preview_enabled": False})
            disabled = await request("POST", f"/v1/cameras/{camera_id}/start")
            await wait_for(lambda: processed_after(15), timeout=45, label="inference without preview encoding")
            assert (await client.get(f"/v1/cameras/{camera_id}/preview.jpg")).status_code == 404
            report["checks"]["preview_disabled_processing_continues"] = True

            await stop_api()
            await command("docker", "stop", args.router_container)
            await start_api()
            # The API can become ready before an optional router starts. Its
            # failed autostart intent must retry after that dependency recovers.
            assert not (await camera())["active"]
            await command("docker", "start", args.router_container)
            restored = await wait_for(lambda: processed_after(8), timeout=50, label="API restart autostart")
            assert restored["session_id"] != disabled["session_id"] and not restored["preview_enabled"]
            old_rows = await request("GET", f"/v1/cameras/{camera_id}/results?limit=500")
            assert len({row["session_id"] for row in old_rows}) >= 3
            report["checks"]["api_restart_identity_history_autostart"] = True
            report["checks"]["autostart_recovers_from_initial_router_outage"] = True
            progress("PASS preview disabled; API restart retains identity/history and autostarts")

            await request("POST", f"/v1/cameras/{camera_id}/stop")
            with av.open(str(args.video.resolve())) as video:
                image = next(video.decode(video=0)).to_ndarray(format="bgr24")
            _, encoded = cv2.imencode(".jpg", image)
            image_upload = await request("POST", "/v1/cameras/media", files={"file": ("driver.jpg", encoded.tobytes(), "image/jpeg")})
            video_upload = await request("POST", "/v1/cameras/media", files={"file": (args.video.name, args.video.read_bytes(), "video/mp4")})
            for kind, upload in (("image", image_upload), ("video", video_upload)):
                identifier = f"{camera_id}-{kind}"
                await request("POST", "/v1/cameras", json={**spec, "camera_id": identifier,
                    "kind": kind, "uri": upload["uri"], "autostart": False})
                await request("POST", f"/v1/cameras/{identifier}/start")
                async def finished(identifier=identifier):
                    row = next(row for row in (await request("GET", "/v1/cameras"))["cameras"] if row["camera_id"] == identifier)
                    assert row.get("status") != "error", row
                    return row if not row["active"] and row.get("processed", 0) > 0 else None
                ended = await wait_for(finished, timeout=50, label=f"{kind} completion")
                history = await request("GET", f"/v1/cameras/{identifier}/results")
                assert history and all(row["camera_id"] == identifier for row in history)
                report["checks"][f"{kind}_input"] = {"processed": ended["processed"], "status": ended["status"]}
                await request("DELETE", f"/v1/cameras/{identifier}")
            progress("PASS uploaded image and video through real models")

            async def evidence_finished():
                rows = await request("GET", "/v1/evidence/records?limit=100")
                return rows if rows and all(row["state"] != "Pending" for row in rows) else None
            evidence = await wait_for(evidence_finished, timeout=75, label="incident evidence finalization")
            assert all(row["state"] in {"Saved", "Incomplete"} for row in evidence), evidence
            assert all("processing_failed" not in row.get("errors", []) for row in evidence), evidence
            assert all(row.get("camera_id", "").startswith(camera_id) for row in evidence)
            clip = next(row for row in evidence if "clip" in row.get("artifacts", {}))
            response = await client.get(f"/v1/evidence/records/{clip['id']}/assets/clip")
            response.raise_for_status()
            with av.open(io.BytesIO(response.content)) as video:
                first = next(video.decode(video=0))
                assert first.width and first.height
            ranged = await client.get(f"/v1/evidence/records/{clip['id']}/assets/clip", headers={"Range": "bytes=0-127"})
            assert ranged.status_code == 206 and len(ranged.content) == 128
            report["checks"]["evidence"] = {"states": dict(Counter(row["state"] for row in evidence)),
                "clip_decoded": True, "http_range": True,
                "incomplete_reasons": sorted({reason for row in evidence if row["state"] == "Incomplete"
                                               for reason in row.get("errors", [])})}
            after = (await client.get("http://127.0.0.1:8000/v2/models/stats")).json() if args.runtime == "triton" else None
            report["triton_before"], report["triton_after"] = before, after
            runs = await request("GET", f"/v1/cameras/{camera_id}/runs")
            await request("DELETE", f"/v1/cameras/{camera_id}")
            assert (await media.get(f"/v3/config/paths/get/inferops/combined/{camera_id}")).status_code == 404
            assert len(await request("GET", f"/v1/cameras/{camera_id}/runs")) == len(runs)
            report["checks"]["remove_cleans_path_preserves_history"] = True
            report["passed"] = True
            progress("PASS evidence clip decoding/range playback and source removal/history")
        finally:
            await stop_api()
            await stop_publisher()
            await command("docker", "start", args.router_container)
            for identifier in (camera_id, camera_id + "-image", camera_id + "-video"):
                with suppress(httpx.HTTPError):
                    await media.delete(f"/v3/config/paths/delete/inferops/combined/{identifier}")
            await command("docker", "rm", "-f", origin)
            await command("playwright-cli", f"-s={browser_session}", "close")
            for file in log_files:
                file.close()
            report["finished_at"] = time.time()
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {root / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cpu", "cuda", "triton"], default="triton")
    parser.add_argument("--video", type=Path, default=ROOT / "research/driver_training/test.mp4")
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--origin-port", type=int, default=19554)
    parser.add_argument("--router", default="http://127.0.0.1:9997")
    parser.add_argument("--rtsp", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--router-container", default="inferops-mediamtx-1")
    parser.add_argument("--router-user", default=os.getenv("INFEROPS_MEDIAMTX_API_USER", "inferops"))
    parser.add_argument("--router-password", default=os.getenv("INFEROPS_MEDIAMTX_API_PASSWORD", "inferops-local-api"))
    parser.add_argument("--output", type=Path, default=ROOT / ".run/verification/media-real")
    parser.add_argument("--database-url", required=True, help="Dedicated disposable PostgreSQL verification database")
    args = parser.parse_args()
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()
