# Driver-face detection and analysis

SCRFD selects the face independently; 6DRepNet360 estimates its head direction,
2D106 locates the eyes, and BlinkCNN classifies their closure. See
[face monitoring](driver-face-models.md) for thresholds and runtime details.

A source's `driver_roi` overrides the detector's `driver_region`. Detection crops
to that region before resizing. Bounded recovery can retry a recent face
neighborhood, context and overlapping crops. A region is important when
passengers are visible; otherwise selection uses the largest face and spatial
continuity. SCRFD confidence remains the measured detection confidence.

Caches are isolated by stream, frame size, selected region/face, and time.
Rewinds and face loss invalidate tracking. Head, landmarks and eye scores have
separate measurement ages. Landmark coordinates return to the original frame
before eye extraction. A different selected face cannot reuse an old eye crop.

`face_detected` and the SCRFD box remain available even when analysis fails.
Missing/hidden eyes produce `unknown`, not `closed`. No RF-DETR behavior box
substitutes for a missing face. Existing temporal incident confirmation remains;
a single blink is not a sustained drowsiness incident.
