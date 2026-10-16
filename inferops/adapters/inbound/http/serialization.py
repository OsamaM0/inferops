"""Live wire serialization without pixels or concrete infrastructure."""
from typing import Any

from inferops.domain.types import StageItem


def live_result(item: StageItem) -> dict[str, Any]:
    """Pre-build the JSON payload by hand so we never serialize numpy frames.

    Domain stages expose extra per-frame state through two generic payload
    channels that the dashboard renders without knowing the domain:

    - ``payload["signals"]``: flat ``{name: number | string}`` shown as chips
    - ``payload["overlay_boxes"]``: ``[{"label", "box": [x1,y1,x2,y2], "tone"}]``
      normalised boxes drawn on the preview in addition to detections
    """
    out: dict[str, Any] = {
        "id": item.id,
        "stream_id": item.stream_id,
        "ts_ms": item.ts_ms,
        "seq": item.seq,
        "results": [],
        "detections": [],
        "classifications": [],
        "fusion_events": item.payload.get("fusion_events", []),
        "alerts": item.payload.get("alerts", []),
        "signals": item.payload.get("signals", {}),
        "overlay_boxes": item.payload.get("overlay_boxes", []),
        "model_selection": item.payload.get("model_selection"),
    }
    last_latency = 0.0
    for r in item.results:
        out["results"].append({
            "kind": r.kind,
            "model": r.model,
            "latency_ms": r.latency_ms,
        })
        last_latency = r.latency_ms or last_latency
        if r.detections:
            for d in r.detections:
                out["detections"].append({
                    "label": d.label,
                    "score": d.score,
                    "box": list(d.box),
                    "track_id": d.attrs.get("track_id"),
                })
        if r.classifications:
            for c in r.classifications:
                out["classifications"].append({
                    "label": c.label, "score": c.score, "index": c.index,
                })
    out["model"] = (item.payload.get("model_selection") or {}).get("used", "")
    out["model_latency_ms"] = last_latency
    return out
