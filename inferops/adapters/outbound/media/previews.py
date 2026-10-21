"""Encode one bounded annotated preview, reusable across all viewers."""

from __future__ import annotations

from inferops.adapters.outbound.media.annotations import _draw_annotations


def render_preview(image, output, spec, annotator=None):
    import cv2

    annotated = _draw_annotations(image, output, annotator)
    h, w = annotated.shape[:2]
    if w > spec.preview_width:
        annotated = cv2.resize(annotated, (spec.preview_width, max(1, round(h * spec.preview_width / w))))
    ok, data = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
    if not ok:
        raise ValueError("preview encoding failed")
    return data.tobytes()
