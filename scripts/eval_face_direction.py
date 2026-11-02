"""Evaluate driver face direction / eye state on real videos.

Runs every video through the ``driver_behaviour`` pipeline, either in-process
(``--mode cpu|cuda|triton``) or against a running server (``--api URL``, frames
are streamed one by one through ``/v1/infer`` exactly like a live camera), and
writes per video:

* ``<video>.csv``          per-frame yaw/pitch/roll, direction, eye state, latency
* ``<video>.annotated.mp4`` overlay for visual review (``--annotate``)
* ``summary.json``         face-found rate, direction histogram, alerts, latency

``--compare A B`` checks two result directories (e.g. cpu vs triton) for
numerical parity of the face signals.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DIRECTIONS = ("front", "left", "right", "up", "down", "away")
FACE_EVENTS = {"looking_left", "looking_right", "looking_up", "looking_down", "eyes_closed",
               "distraction", "drowsiness"}
FIELDS = ("frame", "t_ms", "face", "face_source", "confidence", "yaw", "pitch", "roll",
          "direction", "neutral_yaw", "neutral_pitch", "eye_closure", "look_down", "eye_left", "eye_right",
          "eyes_closed", "perclos",
          "distracted_score", "drowsy_ear", "label", "face_stage_ms", "frame_ms", "events")


def _direction(payload: dict) -> str:
    direction = payload.get("head_direction")
    if direction:
        return str(direction)
    # Pre-upgrade pipelines only expose the human label.
    label = str(payload.get("face_status_label") or "")
    for name in ("Left", "Right", "Up", "Down", "Away"):
        if f"Looking {name}" in label:
            return name.lower()
    return "front" if payload.get("face_box") is not None else ""


def _num(value, digits=2):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return round(float(value), digits)
    return ""


def frame_record(index: int, t_ms: float, payload: dict, frame_ms: float) -> dict:
    pose = payload.get("face_pose") or {}
    events = [e.get("type", "") for e in payload.get("fusion_events") or []]
    events += [a.get("type", "") for a in payload.get("alerts") or []]
    return {
        "frame": index,
        "t_ms": round(t_ms, 1),
        "face": int(payload.get("face_box") is not None),
        "face_source": payload.get("face_source", ""),
        "confidence": _num(payload.get("face_mesh_confidence"), 3),
        "yaw": _num(pose.get("yaw_deg", pose.get("yaw"))),
        "pitch": _num(pose.get("pitch_deg", pose.get("pitch"))),
        "roll": _num(pose.get("roll_deg", pose.get("roll"))),
        "direction": _direction(payload),
        "neutral_yaw": _num((payload.get("neutral_pose") or {}).get("yaw")),
        "neutral_pitch": _num((payload.get("neutral_pose") or {}).get("pitch")),
        "eye_closure": _num(payload.get("eye_closure"), 3),
        "look_down": _num(payload.get("eye_look_down"), 3),
        "eye_left": payload.get("eye_left_state", ""),
        "eye_right": payload.get("eye_right_state", ""),
        "eyes_closed": int(bool(payload.get("eyes_closed"))) if "eyes_closed" in payload else "",
        "perclos": _num(payload.get("perclos"), 3),
        "distracted_score": _num(payload.get("distracted_score"), 3),
        "drowsy_ear": _num(payload.get("drowsy_ear"), 3),
        "label": payload.get("face_status_label", ""),
        "face_stage_ms": _num(payload.get("face_stage_ms")),
        "frame_ms": round(frame_ms, 2),
        "events": "|".join(e for e in events if e),
    }


def annotate(image: np.ndarray, payload: dict, record: dict) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    box = payload.get("face_box")
    direction = record["direction"] or "no face"
    unsafe = direction not in {"front", "no face"} or record["eyes_closed"] == 1
    color = (0, 0, 230) if unsafe else (80, 220, 80)
    if box is not None:
        x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        pose = payload.get("face_pose") or {}
        yaw = pose.get("yaw_deg", pose.get("yaw"))
        pitch = pose.get("pitch_deg", pose.get("pitch"))
        if isinstance(yaw, (int, float)) and isinstance(pitch, (int, float)):
            # Nose-direction arrow: +yaw = driver's left = image right, +pitch = down.
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = max(x2 - x1, 40)
            dx = int(math.sin(math.radians(yaw)) * length)
            dy = int(math.sin(math.radians(pitch)) * length)
            cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy), (255, 200, 0), 3, tipLength=0.25)
        for lm in (payload.get("face_landmarks") or [])[::6]:
            cv2.circle(out, (int(lm["x"] * w), int(lm["y"] * h)), 1, (200, 200, 200), -1)
    lines = [
        f"dir={direction.upper()} yaw={record['yaw']} pitch={record['pitch']} roll={record['roll']}",
        f"eyes L={record['eye_left'] or '-'} R={record['eye_right'] or '-'} closure={record['eye_closure']}"
        f" perclos={record['perclos']}",
        f"src={record['face_source'] or '-'} conf={record['confidence']} face={record['face_stage_ms']}ms"
        f" frame={record['frame_ms']}ms",
    ]
    if record["events"]:
        lines.append("EVENT: " + record["events"])
    for i, text in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if i == 3 else (255, 255, 255), 1, cv2.LINE_AA)
    return out


class VideoReport:
    def __init__(self, video: Path, output: Path, annotate_video: bool):
        self.video = video
        self.rows: list[dict] = []
        self.csv_path = output / f"{video.stem}.csv"
        self.writer = None
        self.annotated = output / f"{video.stem}.annotated.mp4" if annotate_video else None

    def add(self, index: int, t_ms: float, payload: dict, frame_ms: float, image: np.ndarray | None) -> None:
        record = frame_record(index, t_ms, payload, frame_ms)
        self.rows.append(record)
        if self.annotated is not None and image is not None:
            if self.writer is None:
                h, w = image.shape[:2]
                self.writer = cv2.VideoWriter(str(self.annotated), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            self.writer.write(annotate(image, payload, record))

    def finish(self, alerts: list[dict], fusion_events: list[dict], elapsed_s: float) -> dict:
        if self.writer is not None:
            self.writer.release()
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        frames = len(self.rows) or 1
        with_face = [r for r in self.rows if r["face"]]
        face_ms = [r["face_stage_ms"] for r in self.rows if r["face_stage_ms"] != ""]
        frame_ms = [r["frame_ms"] for r in self.rows]
        closed = [r for r in with_face if r["eyes_closed"] == 1]

        def pct(values, q):
            return round(float(np.percentile(values, q)), 2) if values else None

        return {
            "video": self.video.name,
            "frames": len(self.rows),
            "face_found_rate": round(len(with_face) / frames, 3),
            "face_sources": dict(Counter(r["face_source"] for r in with_face)),
            "direction_histogram": dict(Counter(r["direction"] or "no_face" for r in self.rows)),
            "eyes_closed_rate": round(len(closed) / max(len(with_face), 1), 3),
            "alerts_by_type": dict(Counter(a.get("type", "") for a in alerts)),
            "fusion_events_by_type": dict(Counter(e.get("type", "") for e in fusion_events)),
            "face_stage_ms": {"p50": pct(face_ms, 50), "p95": pct(face_ms, 95)},
            "frame_ms": {"p50": pct(frame_ms, 50), "p95": pct(frame_ms, 95),
                         "mean": round(statistics.fmean(frame_ms), 2) if frame_ms else None},
            "throughput_fps": round(len(self.rows) / elapsed_s, 1) if elapsed_s else None,
            "csv": str(self.csv_path),
            "annotated": str(self.annotated) if self.annotated else None,
        }


async def run_in_process(args, videos: list[Path]) -> list[dict]:
    from inferops import Engine

    options = {"triton_url": args.triton_url} if args.mode == "triton" else {}
    engine = Engine.from_project(args.project, runtime=args.mode, **options)
    reports = []
    async with engine:
        application = engine.application
        for video in videos:
            report = VideoReport(video, args.output, args.annotate)
            session = await engine.open_session(args.pipeline, stream_name=f"eval-{video.stem}")
            alerts, events = [], []
            started = time.perf_counter()
            index = 0
            try:
                async for frame in application.source_factory(video, every_n=1):
                    t0 = time.perf_counter()
                    items = await session.process(frame)
                    frame_ms = (time.perf_counter() - t0) * 1000
                    for item in items:
                        payload = item.payload
                        alerts += payload.get("alerts") or []
                        events += payload.get("fusion_events") or []
                        t_ms = float(getattr(frame, "media_time_ns", 0) or 0) / 1e6 or index * 1000 / 15
                        report.add(index, t_ms, payload, frame_ms, payload.get("image_np"))
                        index += 1
                    if args.max_frames and index >= args.max_frames:
                        break
            finally:
                await session.close()
            reports.append(report.finish(alerts, events, time.perf_counter() - started))
            print(json.dumps(reports[-1]), flush=True)
    return reports


def run_api(args, videos: list[Path]) -> list[dict]:
    import httpx

    base = args.api.rstrip("/")
    reports = []
    with httpx.Client(timeout=60) as client:
        for video in videos:
            report = VideoReport(video, args.output, args.annotate)
            opened = client.post(f"{base}/v1/sessions",
                                 json={"pipeline": args.pipeline, "stream_id": f"eval-{video.stem}"})
            opened.raise_for_status()
            session = opened.json()
            headers = {"X-Session-Capability": session["capability"]}
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
            alerts, events = [], []
            started = time.perf_counter()
            index = 0
            try:
                while True:
                    ok, image = capture.read()
                    if not ok or (args.max_frames and index >= args.max_frames):
                        break
                    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                    t_ms = index * 1000.0 / fps
                    t0 = time.perf_counter()
                    response = client.post(
                        f"{base}/v1/infer/{args.pipeline}",
                        headers=headers,
                        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                        data={"session_id": session["session_id"], "seq": str(index),
                              "media_time_ns": str(int(t_ms * 1e6))},
                    )
                    response.raise_for_status()
                    frame_ms = (time.perf_counter() - t0) * 1000
                    for item in response.json():
                        payload = item.get("payload") or {}
                        alerts += payload.get("alerts") or []
                        events += payload.get("fusion_events") or []
                        report.add(index, t_ms, payload, frame_ms, image)
                    index += 1
            finally:
                capture.release()
                client.delete(f"{base}/v1/sessions/{session['session_id']}", headers=headers)
            reports.append(report.finish(alerts, events, time.perf_counter() - started))
            print(json.dumps(reports[-1]), flush=True)
    return reports


def compare(first: Path, second: Path, max_angle_deg: float, min_agreement: float) -> dict:
    """Compare two runs over the same videos (e.g. cpu vs triton).

    Alerts must match exactly; per-frame states and angles within tolerance.
    Landmark tracking feeds each frame's crop from the last landmarks, so tiny
    float differences between ONNX Runtime builds grow on very small faces.
    """
    result = {"first": str(first), "second": str(second), "videos": {}, "passed": True}

    def alerts(run: Path) -> dict[str, dict]:
        path = run / "summary.json"
        videos = json.loads(path.read_text())["videos"] if path.is_file() else []
        return {Path(v["video"]).stem: v.get("alerts_by_type", {}) for v in videos}

    left_alerts, right_alerts = alerts(first), alerts(second)
    for left_csv in sorted(first.glob("*.csv")):
        right_csv = second / left_csv.name
        if not right_csv.is_file():
            continue
        left = {int(r["frame"]): r for r in csv.DictReader(left_csv.open())}
        right = {int(r["frame"]): r for r in csv.DictReader(right_csv.open())}
        shared = sorted(set(left) & set(right))
        diffs, agree, both = [], 0, 0
        for frame in shared:
            a, b = left[frame], right[frame]
            agree += a["direction"] == b["direction"] and a["eyes_closed"] == b["eyes_closed"]
            if a["yaw"] and b["yaw"]:
                both += 1
                diffs.append(max(abs(float(a["yaw"]) - float(b["yaw"])),
                                 abs(float(a["pitch"]) - float(b["pitch"]))))
        stats = {
            "frames": len(shared),
            "state_agreement": round(agree / max(len(shared), 1), 4),
            "angle_diff_p95_deg": round(float(np.percentile(diffs, 95)), 3) if diffs else None,
            "angle_diff_max_deg": round(max(diffs), 3) if diffs else None,
        }
        stats["alerts"] = {"first": left_alerts.get(left_csv.stem), "second": right_alerts.get(left_csv.stem)}
        stats["alerts_match"] = stats["alerts"]["first"] == stats["alerts"]["second"]
        stats["passed"] = stats["alerts_match"] and stats["state_agreement"] >= min_agreement and (
            stats["angle_diff_p95_deg"] is None or stats["angle_diff_p95_deg"] <= max_angle_deg)
        result["passed"] &= stats["passed"]
        result["videos"][left_csv.stem] = stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["cpu", "cuda", "tensorrt", "triton"], default="cuda")
    parser.add_argument("--api", help="Base URL of a running server, e.g. https://<pod>-8082.proxy.runpod.net")
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument("--project", type=Path, default=Path("projects/driver_behaviour"))
    parser.add_argument("--pipeline", default="driver_behaviour")
    parser.add_argument("--videos", type=Path, nargs="*",
                        default=sorted(Path("research/driver_training").glob("*.mp4")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--max-angle-diff", type=float, default=1.0)
    parser.add_argument("--min-agreement", type=float, default=0.99)
    args = parser.parse_args()

    if args.compare:
        report = compare(*args.compare, args.max_angle_diff, args.min_agreement)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)

    label = "api" if args.api else args.mode
    args.output = (args.output or Path(".run/eval_face_direction") / label).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.videos:
        parser.error("no videos found")
    reports = run_api(args, args.videos) if args.api else asyncio.run(run_in_process(args, args.videos))
    summary = {"mode": label, "api": args.api, "pipeline": args.pipeline, "videos": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
