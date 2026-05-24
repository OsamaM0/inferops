"""Generic preprocess stage: dispatches to the model's preprocess plugin."""
from __future__ import annotations

from inferops.core.plugins import instantiate
from inferops.core.types import Frame, StageItem
from inferops.pipeline.stage import Stage


class PreprocessGeneric(Stage):
    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self.model = model
        self._cache = {}

    async def _get_pre(self, spec):
        if spec.name in self._cache:
            return self._cache[spec.name]
        if spec.preprocess is None:
            return None
        obj = instantiate(spec.preprocess.plugin, spec.preprocess.params)
        self._cache[spec.name] = obj
        return obj

    async def process(self, item: StageItem):
        reg = self.ctx.registry
        if reg is None:
            return item
        model_name = self.model or item.payload.get("model")
        if not model_name:
            return item
        spec = reg.get_spec(model_name)
        pre = await self._get_pre(spec)
        if pre is None:
            return item
        frame = Frame(
            stream_id=item.stream_id, ts_ms=item.ts_ms, seq=item.seq,
            image=item.payload.get("image_np"),
            width=item.payload.get("width", 0), height=item.payload.get("height", 0),
            meta=item.payload.setdefault("frame_meta", {}),
        )
        feeds = await pre(frame, spec)
        item.payload["feeds"] = feeds
        item.payload["frame_meta"] = frame.meta
        return item
