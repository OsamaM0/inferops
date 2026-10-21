"""Small Control API adapter; MediaMTX owns routing and its browser player."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

import httpx

from inferops.domain.errors import BackendError


class MediaRelayError(BackendError):
    """A safe public error, without upstream URLs or camera credentials."""


class MediaMTX:
    def __init__(self, api_url, rtsp_url, *, playback_url="http://127.0.0.1:9996", record=True,
                 scope="combined", username="inferops", password="", transport=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", scope):
            raise ValueError("invalid media relay scope")
        for value, schemes in ((api_url, {"http", "https"}), (playback_url, {"http", "https"}),
                               (rtsp_url, {"rtsp", "rtsps"})):
            parsed = urlsplit(value)
            if (parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment):
                raise ValueError("media relay endpoints must have the expected scheme and no credentials/query")
        self.rtsp_url, self.scope = rtsp_url.rstrip("/"), scope
        self.record = record
        self.client = httpx.AsyncClient(base_url=api_url.rstrip("/"), timeout=5, trust_env=False,
                                       auth=(username, password), transport=transport)
        self.playback = httpx.AsyncClient(base_url=playback_url.rstrip("/"), timeout=10, trust_env=False,
                                         auth=(username, password), transport=transport)

    def path(self, camera_id):
        # Scope isolates static workers sharing a single MediaMTX instance.
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", camera_id):
            raise ValueError("invalid camera ID")
        return f"inferops/{self.scope}/{camera_id}"

    async def _request(self, method, path, **kwargs):
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.HTTPError:
            raise MediaRelayError("MediaMTX is unavailable; check the media service and internal API settings") from None
        if response.status_code != 200 and not (response.status_code == 404 and method in {"GET", "DELETE"}):
            raise MediaRelayError("MediaMTX rejected the operation; check its API authentication and server logs")
        return response

    async def prepare(self, spec):
        if spec.kind != "rtsp":
            return spec
        name = self.path(spec.camera_id)
        config = {"source": spec.uri, "sourceOnDemand": True, "rtspTransport": spec.rtsp_transport,
                  "sourceOnDemandStartTimeout": f"{max(10, spec.timeout_s):g}s",
                  "sourceOnDemandCloseAfter": "5s", "record": self.record, "alwaysAvailable": False}
        current = await self._request("GET", f"/v3/config/paths/get/{name}")
        if current.status_code == 404:
            await self._request("POST", f"/v3/config/paths/add/{name}", json=config)
        else:
            try:
                existing = current.json()
                changed = any(existing.get(key) != value for key, value in config.items())
            except (ValueError, AttributeError):
                raise MediaRelayError("MediaMTX returned an invalid path configuration") from None
            if changed:
                await self._request("PATCH", f"/v3/config/paths/patch/{name}", json=config)
        # One shared compressed upstream. PyAV still decodes and samples locally.
        return spec.model_copy(update={"uri": f"{self.rtsp_url}/{name}", "rtsp_transport": "tcp"})

    async def remove(self, camera_id):
        await self._request("DELETE", f"/v3/config/paths/delete/{self.path(camera_id)}")

    async def recordings(self, camera_id, *, start=None, end=None):
        params = {"path": self.path(camera_id)}
        params.update({key: value for key, value in (("start", start), ("end", end)) if value is not None})
        try:
            response = await self.playback.get("/list", params=params)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("invalid timespans")
            # Never expose an internal hostname, source URI or credentials.
            return [{"start": row["start"], "duration": float(row["duration"])} for row in rows]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise MediaRelayError("MediaMTX recording playback is unavailable; check recording and playback settings") from None

    async def status(self):
        try:
            response = await self._request("GET", "/v3/paths/list", params={"itemsPerPage": 1})
            return {"enabled": True, "status": response.status_code, "recording": self.record,
                    "features": ["rtsp_relay", "on_demand", "webrtc", "hls", "recording", "playback", "metrics"]}
        except MediaRelayError:
            return {"enabled": True, "status": 503}

    async def close(self):
        await self.client.aclose()
        await self.playback.aclose()


def configured_relay(*, scope="combined"):
    selected = os.getenv("INFEROPS_MEDIA_ROUTER", "mediamtx").strip().lower()
    if selected == "direct":
        return None
    if selected != "mediamtx":
        raise ValueError("INFEROPS_MEDIA_ROUTER must be direct or mediamtx")
    return MediaMTX(os.getenv("INFEROPS_MEDIAMTX_API_URL", "http://127.0.0.1:9997"),
                    os.getenv("INFEROPS_MEDIAMTX_RTSP_URL", "rtsp://127.0.0.1:8554"), scope=scope,
                    playback_url=os.getenv("INFEROPS_MEDIAMTX_PLAYBACK_URL", "http://127.0.0.1:9996"),
                    record=os.getenv("INFEROPS_MEDIAMTX_RECORD", "true").strip().lower() in {"true", "1", "yes"},
                    username=os.getenv("INFEROPS_MEDIAMTX_API_USER", "inferops"),
                    password=os.getenv("INFEROPS_MEDIAMTX_API_PASSWORD", "inferops-local-api"))
