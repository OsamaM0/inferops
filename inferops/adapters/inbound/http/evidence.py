"""Evidence APIs and a compatibility link to the original MinIO Console."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from inferops.adapters.inbound.http.ui import service_url

router = APIRouter()

@router.get("/evidence/file/{bucket}/{key:path}")
async def serve_file(bucket: str, key: str, request: Request):
    """Proxy a file from MinIO with HTTP Range support (required for HTML5 video)."""
    storage = request.app.state.engine.application.evidence.storage
    if not hasattr(storage, "client"):
        raise HTTPException(409, "This compatibility route requires object storage; use evidence record assets")
    await storage.start()
    client = storage.client
    # HEAD first to get total size and content-type
    try:
        head = await asyncio.to_thread(
            client.head_object, Bucket=bucket, Key=key
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    total_size: int = head["ContentLength"]
    content_type: str = head.get("ContentType", "application/octet-stream")
    filename = key.split("/")[-1]
    inline_types = ("image/", "video/")
    disposition = (
        "inline"
        if any(content_type.startswith(t) for t in inline_types)
        else f'attachment; filename="{filename}"'
    )

    range_header = request.headers.get("range")
    if range_header:
        # Parse  "bytes=<start>-<end>"
        try:
            range_val = range_header.strip().removeprefix("bytes=")
            raw_start, _, raw_end = range_val.partition("-")
            start = int(raw_start) if raw_start else 0
            end = int(raw_end) if raw_end else total_size - 1
        except ValueError as e:
            raise HTTPException(status_code=416, detail="Invalid Range header") from e

        end = min(end, total_size - 1)
        if start > end:
            raise HTTPException(
                status_code=416,
                detail=f"Range not satisfiable: {start}-{end}/{total_size}",
            )
        length = end - start + 1

        try:
            obj = await asyncio.to_thread(
                client.get_object,
                Bucket=bucket, Key=key,
                Range=f"bytes={start}-{end}",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        body = obj["Body"]

        def _iter_range():
            try:
                while chunk := body.read(65536):
                    yield chunk
            finally:
                body.close()

        return StreamingResponse(
            _iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Disposition": disposition,
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    # Full file (no Range header)
    try:
        obj = await asyncio.to_thread(
            client.get_object, Bucket=bucket, Key=key
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    body = obj["Body"]

    def _iter():
        try:
            while chunk := body.read(65536):
                yield chunk
        finally:
            body.close()

    return StreamingResponse(
        _iter(),
        media_type=content_type,
        headers={
            "Content-Disposition": disposition,
            "Accept-Ranges": "bytes",
            "Content-Length": str(total_size),
        },
    )


@router.get("/evidence/clips", response_class=JSONResponse, tags=["evidence"])
async def list_clips(request: Request, limit: int = 100):
    """List persisted clips from the authoritative evidence records."""
    if not 1 <= limit <= 1000:
        raise HTTPException(400, "limit must be between 1 and 1000")
    records = await request.app.state.engine.application.repository.list_evidence(limit=limit)
    items = []
    for record in records:
        clip = (record.get("artifacts") or {}).get("clip")
        if clip:
            items.append({"key": clip["uri"], "size_bytes": clip.get("size_bytes"),
                          "last_modified": record.get("updated_at"), "type": "mp4",
                          "url": f"/v1/evidence/records/{record['id']}/assets/clip"})
    return {"count": len(items), "clips": items}


@router.get("/evidence", tags=["evidence"])
async def evidence_browser(request: Request):
    """Open the original MinIO Console; object listing and file APIs remain available."""
    return RedirectResponse(service_url(request, "minio"), status_code=307)
