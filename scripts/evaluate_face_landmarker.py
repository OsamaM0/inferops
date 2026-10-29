"""Replay a face pipeline without HTTP, source state, alerts, or database writes."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.stages.base import StageContext
from inferops.application.models import ModelPool
from inferops.domain.types import StageItem


async def evaluate(args):
    definition = ProjectLoader(args.project.resolve(), mode=args.mode).load()
    registry = ModelPool(definition)
    stages, videos, panels = [], [], []
    rois = json.loads(args.rois.read_text()) if args.rois else {}
    try:
        for declaration in definition.pipelines['driver_face'].stages:
            if ':' not in declaration.stage:
                continue
            stage = definition.object_loader(declaration.stage)(**declaration.params)
            await stage.setup(StageContext(pipeline='face_replay', name=declaration.name, registry=registry))
            stages.append(stage)
        for path in sorted(args.videos.glob('*.mp4')):
            cap = cv2.VideoCapture(str(path))
            rows = []
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps <= 0 or total <= 0:
                    raise ValueError(f'Cannot read video metadata: {path}')
                indices = np.unique(np.linspace(0, total - 1, args.samples, dtype=int))
                snapshots = set(indices[np.linspace(0, len(indices) - 1, 4, dtype=int)].tolist())
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, image = cap.read()
                    if not ok:
                        continue
                    payload = {'image_np': image}
                    if path.name in rois:
                        payload['driver_roi'] = rois[path.name]
                    item = StageItem(stream_id=path.stem, ts_ms=int(idx / fps * 1000), payload=payload)
                    start = time.perf_counter()
                    for stage in stages:
                        item = await stage.process(item)
                    rows.append({'frame': int(idx), 'ts_ms': item.ts_ms,
                        'latency_ms': (time.perf_counter() - start) * 1000,
                        'face': bool(item.payload.get('face_box')), 'signals': item.payload.get('signals', {}),
                        'pose': item.payload.get('face_pose'), 'box': item.payload.get('face_box')})
                    if int(idx) in snapshots:
                        panels.append(panel(image, path.name, rows[-1]))
            finally:
                cap.release()
            latency = [r['latency_ms'] for r in rows]
            result = {'video': path.name, 'frames': len(rows), 'faces': sum(r['face'] for r in rows),
                'mean_ms': float(np.mean(latency)), 'p50_ms': float(np.median(latency)),
                'p95_ms': float(np.percentile(latency, 95)), 'driver_roi': rois.get(path.name),
                'eyes': dict(Counter(r['signals'].get('eyes_state', 'unknown') for r in rows)),
                'directions': dict(Counter(r['signals'].get('head_direction', 'unknown') for r in rows)),
                'rows': rows}
            videos.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    finally:
        for stage in reversed(stages):
            await stage.teardown()
        await registry.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(videos, indent=2) + '\n')
    if panels:
        while len(panels) % 4:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)])
        cv2.imwrite(str(args.output.with_suffix('.jpg')), sheet)


def panel(image, name, row):
    image = image.copy()
    h, w = image.shape[:2]
    if row['box']:
        x1, y1, x2, y2 = [int(v * size) for v, size in zip(row['box'], (w, h, w, h), strict=True)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), max(1, w // 300))
    canvas = np.zeros((300, 360, 3), np.uint8)
    scale = min(360 / w, 240 / h)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(image, size)
    x, y = (360 - size[0]) // 2, (240 - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    signals = row['signals']
    for y, text in [(257, name[:42]), (276, f"frame {row['frame']} | head {signals.get('head_direction', 'unknown')}"),
                    (294, f"eyes {signals.get('eyes_state', 'unknown')}")]:
        cv2.putText(canvas, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('projects/driver_behaviour'))
    parser.add_argument('--videos', type=Path, default=Path('research/driver_training'))
    parser.add_argument('--mode', default='cpu', choices=['cpu', 'cuda', 'triton', 'tensorrt'])
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--rois', type=Path, help='JSON mapping video filename to normalized driver ROI')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    asyncio.run(evaluate(args))
