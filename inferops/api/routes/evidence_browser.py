"""Evidence APIs and a compatibility link to the original MinIO Console."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from inferops.adapters.s3_client import s3_client
from inferops.api.ui import service_url

router = APIRouter()

CLIPS_BUCKET = "inferops-clips"


def _list_objects(bucket: str, prefix: str = "") -> list[dict]:
    try:
        paginator = s3_client().get_paginator("list_objects_v2")
        objs = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objs.append(obj)
        return sorted(objs, key=lambda o: o["LastModified"], reverse=True)
    except Exception:
        return []


def _file_url(bucket: str, key: str) -> str:
    """Internal proxy URL — slashes kept so FastAPI {key:path} routes correctly."""
    return f"/v1/evidence/file/{bucket}/{key}"


@router.get("/evidence/file/{bucket}/{key:path}")
async def serve_file(bucket: str, key: str, request: Request):
    """Proxy a file from MinIO with HTTP Range support (required for HTML5 video)."""
    # HEAD first to get total size and content-type
    try:
        head = await asyncio.to_thread(
            s3_client().head_object, Bucket=bucket, Key=key
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
                s3_client().get_object,
                Bucket=bucket, Key=key,
                Range=f"bytes={start}-{end}",
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        body = obj["Body"]

        def _iter_range():
            while chunk := body.read(65536):
                yield chunk

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
            s3_client().get_object, Bucket=bucket, Key=key
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    body = obj["Body"]

    def _iter():
        while chunk := body.read(65536):
            yield chunk

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
async def list_clips(limit: int = 100):
    """List alert clips with proxy download URLs."""
    objects = await asyncio.to_thread(_list_objects, CLIPS_BUCKET)
    items = []
    for obj in objects[:limit]:
        key: str = obj["Key"]
        items.append({
            "key": key,
            "size_bytes": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
            "type": key.rsplit(".", 1)[-1].lower(),
            "url": _file_url(CLIPS_BUCKET, key),
        })
    return {"count": len(items), "clips": items}


@router.get("/evidence", tags=["evidence"])
async def evidence_browser(request: Request):
    """Open the original MinIO Console; object listing and file APIs remain available."""
    return RedirectResponse(service_url(request, "minio"), status_code=307)
