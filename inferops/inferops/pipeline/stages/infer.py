"""Inference stage: combines preprocess (if not already done) + backend + postprocess."""
from __future__ import annotations

import time

from inferops.core.plugins import instantiate
from inferops.core.types import Frame, StageItem
from inferops.pipeline.stage import Stage


class Infer(Stage):
    def __init__(self, model: str, fallback_model: str | None = None) -> None:
        super().__init__(model=model, fallback_model=fallback_model)
        self.model = model
        self.fallback_model = fallback_model
        self._resources: dict[str, dict] = {}
        self._active_model = model

    async def setup(self, ctx):
        await super().setup(ctx)
        for model_name in [self.model, self.fallback_model]:
            if not model_name or model_name in self._resources:
                continue
            spec = ctx.registry.get_spec(model_name)
            self._resources[model_name] = {
                "spec": spec,
                "pre": instantiate(spec.preprocess.plugin, spec.preprocess.params) if spec.preprocess else None,
                "post": instantiate(spec.postprocess.plugin, spec.postprocess.params) if spec.postprocess else None,
            }

    async def _run_with_model(self, model_name: str, item: StageItem, frame: Frame):
        reg = self.ctx.registry
        bundle = self._resources[model_name]
        feeds = item.payload.get("feeds")
        if feeds is None and bundle["pre"] is not None:
            feeds = await bundle["pre"](frame, bundle["spec"])
            item.payload["frame_meta"] = frame.meta
        if feeds is None:
            return item

        backend = await reg.get_backend(model_name)
        t0 = time.perf_counter()
        raw = await backend.infer(feeds)
        latency_ms = (time.perf_counter() - t0) * 1000
        if bundle["post"] is not None:
            result = await bundle["post"](raw, frame, bundle["spec"])
            result.latency_ms = latency_ms
            item.results.append(result)
        else:
            from inferops.core.types import Result
            item.results.append(Result(kind="raw",
                                       raw={k: v.tolist() for k, v in raw.items()},
                                       model=model_name, latency_ms=latency_ms))
        item.payload["model_selection"] = {"requested": self.model, "used": model_name}
        item.payload.pop("feeds", None)
        self._active_model = model_name
        return item

    async def process(self, item: StageItem):
        frame = Frame(
            stream_id=item.stream_id, ts_ms=item.ts_ms, seq=item.seq,
            image=item.payload.get("image_np"),
            width=item.payload.get("width", 0), height=item.payload.get("height", 0),
            meta=item.payload.setdefault("frame_meta", {}),
        )
        candidates = [self._active_model]
        if self._active_model != self.model:
            return await self._run_with_model(self._active_model, item, frame)
        if self.fallback_model:
            candidates.append(self.fallback_model)

        last_error = None
        for model_name in candidates:
            try:
                return await self._run_with_model(model_name, item, frame)
            except Exception as exc:
                last_error = exc
                if model_name != self.model or not self.fallback_model:
                    raise

        if last_error is not None:
            raise last_error
        return item
