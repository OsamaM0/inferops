"""Verify real CPU/CUDA inference, persisted alerts, WS fanout and MinIO media."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import time
import uuid
from pathlib import Path

import av
import boto3
import cv2
import httpx
import websockets


async def verify(args) -> dict:
    video = args.video.resolve()
    cap = cv2.VideoCapture(str(video))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert expected_frames > 0, "video is not readable"
    stream_id = f"verify-{args.mode}-{uuid.uuid4().hex[:8]}"
    received = []
    socket_url = args.api.replace("http://", "ws://").replace("https://", "wss://")
    async with httpx.AsyncClient(base_url=args.api, timeout=600) as client:
        async with websockets.connect(f"{socket_url}/v1/ws/alerts.{stream_id}") as ws:
            async def collect():
                async for raw in ws:
                    received.append(json.loads(raw))

            listener = asyncio.create_task(collect())
            try:
                start = time.perf_counter()
                response = await client.post(f"/v1/run/{args.pipeline}/upload",
                    data={"stream_id": stream_id}, files={"video": (video.name, video.read_bytes(), "video/mp4")})
                elapsed = time.perf_counter() - start
                response.raise_for_status()
                summary = response.json()
                status = (await client.get("/v1/system/status")).json()
                alerts = (await client.get("/v1/alerts?limit=1000")).json()
                events = (await client.get("/v1/events?limit=1000")).json()
                await asyncio.sleep(0.1)
            finally:
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)

        assert summary["frames"] == expected_frames, summary
        assert summary["execution_mode"] == args.mode
        assert summary["models_used"] == ["rfdetr_driver_onnx"], "real model must serve every frame"
        assert summary["alerts"], "this sample should exercise alerts"
        emitted_ids = {a["id"] for a in summary["alerts"]}
        assert emitted_ids <= {a["id"] for a in received}, "WS missed alerts"
        persisted_alerts = [a for a in alerts if a.get("entity_id") == stream_id]
        persisted_events = [e for e in events if e.get("entity_id") == stream_id]
        assert emitted_ids <= {a["id"] for a in persisted_alerts}, "alerts missing from Postgres"
        assert persisted_events, "fusion events missing from Postgres"
        runtime_models = {m["model"]: m for m in status["runtime"]["models"]}
        device = "cpu" if args.mode == "cpu" else "gpu"
        for name in ["rfdetr_driver_onnx", "face_mesh_onnx"]:
            assert runtime_models[name]["device"] == device, runtime_models[name]
            assert not runtime_models[name]["downgraded"], runtime_models[name]

        s3 = boto3.client("s3", endpoint_url=args.s3_endpoint,
                          aws_access_key_id=os.getenv("INFEROPS_S3_ACCESS_KEY", "minio"),
                          aws_secret_access_key=os.getenv("INFEROPS_S3_SECRET_KEY", "minio12345"))
        evidence = []
        metadata = []
        for uri in summary["evidence_uris"]:
            bucket, key = uri.removeprefix("s3://").split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            assert data, uri
            record = {"uri": uri, "bytes": len(data), "content_type": obj["ContentType"]}
            if key.endswith(".json"):
                meta = json.loads(data)
                assert meta["stream_id"] == stream_id
                assert all(r["model"] == "rfdetr_driver_onnx" for r in meta["results"])
                metadata.append(meta)
            elif key.endswith(".mp4"):
                with av.open(io.BytesIO(data)) as container:
                    track = container.streams.video[0]
                    record["codec"] = track.codec_context.name
                    record["fps"] = float(track.average_rate)
                    record["frames"] = len(list(container.decode(video=0)))
                assert record["codec"] == "h264"
                assert abs(record["fps"] - source_fps) < 0.01
                assert record["frames"] > 0
                media_response = await client.get(f"/v1/evidence/file/{bucket}/{key}", headers={"Range": "bytes=0-127"})
                assert media_response.status_code == 206
                assert len(media_response.content) == 128
            elif key.endswith(".jpg"):
                import numpy as np
                assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None
            evidence.append(record)
        assert len(metadata) == expected_frames
        assert {m["seq"] for m in metadata} == set(range(1, expected_frames + 1))
        assert any(o["uri"].endswith(".mp4") for o in evidence), "alert clips missing"
        assert any(o["uri"].endswith(".jpg") for o in evidence), "trigger snapshots missing"
        assert all(a.get("payload", {}).get("evidence_uris") for a in persisted_alerts), "alerts missing evidence links"
        return {"passed": True, "mode": args.mode, "video": str(video), "stream_id": stream_id,
                "wall_seconds": round(elapsed, 3), "source_fps": source_fps,
                "summary": summary, "runtime": status["runtime"],
                "websocket_alerts": received, "persisted_alerts": persisted_alerts,
                "persisted_events": persisted_events, "evidence_objects": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--mode", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--pipeline", default="driver_behaviour")
    parser.add_argument("--s3-endpoint", default=os.getenv("INFEROPS_S3_ENDPOINT", "http://localhost:9000"))
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parents[2] / "research/driver_training/test.mp4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(verify(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    summary = result["summary"]
    clips = sum(o["uri"].endswith(".mp4") for o in result["evidence_objects"])
    print(f"PASS {args.mode}: {summary['frames']} frames; {len(summary['alerts'])} alerts; "
          f"{len(result['evidence_objects'])} MinIO objects ({clips} clips); {result['wall_seconds']}s")


if __name__ == "__main__":
    main()
