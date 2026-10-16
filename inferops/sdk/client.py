"""Python SDK for the InferOps REST API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class Client:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0) -> None:
        self.base = base_url.rstrip("/")
        self._c = httpx.AsyncClient(base_url=self.base, timeout=timeout)

    async def close(self) -> None:
        await self._c.aclose()

    async def health(self) -> dict[str, Any]:
        return (await self._c.get("/healthz")).json()

    async def list_models(self) -> list[dict]:
        return (await self._c.get("/v1/models")).json()

    async def list_pipelines(self) -> list[dict]:
        return (await self._c.get("/v1/pipelines")).json()

    async def infer(self, pipeline: str, image: str | bytes | Path,
                    stream_id: str = "sdk", *, session_id: str | None = None,
                    capability: str | None = None, media_time_ns: int | None = None,
                    seq: int | None = None) -> list[dict]:
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        files = {"image": ("input.jpg", image, "image/jpeg")}
        data = {"stream_id": stream_id}
        data.update({name: str(value) for name, value in
                     (("session_id", session_id), ("media_time_ns", media_time_ns), ("seq", seq)) if value is not None})
        headers = {"X-Session-Capability": capability} if capability else {}
        r = await self._c.post(f"/v1/infer/{pipeline}", files=files, data=data, headers=headers)
        r.raise_for_status()
        return r.json()

    async def open_session(self, pipeline: str, stream_id: str = "sdk") -> dict:
        response = await self._c.post("/v1/sessions", json={"pipeline": pipeline, "stream_id": stream_id})
        response.raise_for_status()
        return response.json()

    async def close_session(self, session_id: str, capability: str) -> dict:
        response = await self._c.delete(f"/v1/sessions/{session_id}", headers={"X-Session-Capability": capability})
        response.raise_for_status()
        return response.json()
