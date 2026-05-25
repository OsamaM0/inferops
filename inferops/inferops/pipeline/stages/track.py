"""Track stage — assigns stable `track_id` to detections.

This stage is **auto-injected** by the pipeline runner when the YAML config
contains a `tracking:` block with `enabled: true`. It can also be added
manually as a regular stage entry (`stage: track`).
"""
from __future__ import annotations

from inferops.core.tracking import IoUKalmanTracker, Tracker
from inferops.core.types import StageItem
from inferops.obs import TRACKS_ACTIVE, TRACKS_CLOSED, TRACKS_STARTED, get_logger
from inferops.pipeline.stage import Stage

log = get_logger(__name__)


class Track(Stage):
    """Multi-object tracker stage.

    Parameters mirror `core.tracking.TrackingSpec`. One Tracker instance is
    maintained per scope key (`by`).
    """

    def __init__(
        self,
        *,
        algorithm: str = "iou_kalman",
        by: str = "stream_id",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        low_threshold: float = 0.5,
        score_high: float = 0.5,
        score_low: float = 0.1,
        per_class: bool = True,
        model_filter: list[str] | None = None,
    ) -> None:
        super().__init__(
            algorithm=algorithm, by=by, max_age=max_age, min_hits=min_hits,
            iou_threshold=iou_threshold, low_threshold=low_threshold,
            score_high=score_high, score_low=score_low, per_class=per_class,
            model_filter=model_filter or [],
        )
        self._algorithm = algorithm
        self._by = by
        self._tracker_kwargs = dict(
            max_age=max_age, min_hits=min_hits,
            iou_threshold=iou_threshold, low_threshold=low_threshold,
            score_high=score_high, score_low=score_low, per_class=per_class,
        )
        self._model_filter: set[str] = set(model_filter or [])
        self._trackers: dict[str, Tracker] = {}

    def _tracker_for(self, item: StageItem) -> Tracker:
        if self._by == "stream_id":
            key = item.stream_id or "_"
        elif self._by == "pipeline":
            key = self.ctx.pipeline
        else:
            key = "_global"
        tr = self._trackers.get(key)
        if tr is None:
            tr = IoUKalmanTracker(**self._tracker_kwargs)
            self._trackers[key] = tr
        return tr

    async def process(self, item: StageItem):
        if not item.results:
            return item
        tracker = self._tracker_for(item)
        pipeline = self.ctx.pipeline
        started_before = sum(1 for r in item.results for d in (r.detections or [])
                             if d.attrs.get("track_id"))
        per_label_started: dict[str, int] = {}
        for result in item.results:
            if not result.detections:
                continue
            if self._model_filter and result.model not in self._model_filter:
                continue
            tracker.update(list(result.detections))
        # Observability
        new_ids = sum(1 for r in item.results for d in (r.detections or [])
                      if d.attrs.get("track_id")) - started_before
        if new_ids > 0:
            # Aggregate by label for the started counter.
            for r in item.results:
                if not r.detections:
                    continue
                for d in r.detections:
                    if d.attrs.get("track_id"):
                        per_label_started[d.label] = per_label_started.get(d.label, 0) + 1
            # `started` here is approximate (counts re-emitted IDs too); the
            # tracker itself owns the canonical start/close lifecycle.
        try:
            closed = getattr(tracker, "pop_closed_ids", lambda: [])()
        except Exception:
            closed = []
        if closed:
            TRACKS_CLOSED.labels(pipeline=pipeline, label="*").inc(len(closed))
        active = getattr(tracker, "active_count", 0)
        TRACKS_ACTIVE.labels(pipeline=pipeline, label="*").set(active)
        if per_label_started:
            for lbl, n in per_label_started.items():
                TRACKS_STARTED.labels(pipeline=pipeline, label=lbl).inc(n)
        return item
