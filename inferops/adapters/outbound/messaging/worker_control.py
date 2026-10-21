"""Authenticated internal controls with bounded RPC and preview buffering."""
import httpx
from fastapi import HTTPException
from fastapi.responses import Response


class HTTPWorkerControl:
    def __init__(self, worker, token, *, transport=None):
        self.spec = worker
        self.client = httpx.AsyncClient(base_url=worker.endpoint.rstrip("/"), timeout=10,
                                       headers={"X-Worker-Token": token}, transport=transport)
        self.cached = []

    async def forward(self, method, path, *, body=None, params=None):
        try:
            async with self.client.stream(method, path, json=body, params=params) as response:
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 8 * 1024**2:
                        raise HTTPException(502, "worker response exceeds control buffer limit")
                return Response(bytes(data), status_code=response.status_code,
                                media_type=response.headers.get("content-type", "application/json"),
                                headers={"Cache-Control": "no-store"})
        except httpx.HTTPError as error:
            raise HTTPException(503, f"worker {self.spec.worker_id} is unavailable") from error

    async def status(self):
        try:
            response = await self.client.get("/v1/cameras")
            response.raise_for_status()
            result = response.json()
            self.cached = result["cameras"]
            return {**result, "worker_id": self.spec.worker_id, "available": True}
        except (httpx.HTTPError, ValueError, KeyError):
            return {"worker_id": self.spec.worker_id, "available": False, "active": 0,
                    "cameras": [{**row, "status": "worker_unavailable", "active": False} for row in self.cached]}

    async def close(self):
        await self.client.aclose()
