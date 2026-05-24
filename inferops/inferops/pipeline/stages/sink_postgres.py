"""Persist each StageItem into the generic `events` table."""
from __future__ import annotations

from inferops.adapters.postgres_dao import insert_event
from inferops.core.types import new_id
from inferops.pipeline.stage import SinkStage


class SinkPostgres(SinkStage):
    def __init__(self, type_field: str = "event") -> None:
        super().__init__(type_field=type_field)
        self.type_field = type_field

    async def process(self, item):
        payload = {
            k: v for k, v in item.payload.items()
            if k not in {"image_np", "image_bytes", "feeds", "frame_meta"}
        }
        await insert_event({
            "id": new_id(),
            "pipeline": self.ctx.pipeline,
            "type": self.type_field,
            "entity_id": item.stream_id or "",
            "ts_ms": item.ts_ms,
            "payload": {
                "id": item.id,
                "stream_id": item.stream_id,
                "ts_ms": item.ts_ms,
                "seq": item.seq,
                "frame_ref": item.frame_ref,
                "payload": payload,
                "results": [r.model_dump(mode="json") for r in item.results],
            },
        })
        return None
