"""Evidence browser — lists MinIO objects and streams them via the API."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from inferops.adapters.s3_client import s3_client
from inferops.core.config import settings

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


@router.get("/evidence", response_class=HTMLResponse, tags=["evidence"])
async def evidence_browser():
    """HTML evidence browser showing alert clips from MinIO."""
    objects = await asyncio.to_thread(_list_objects, CLIPS_BUCKET)
    json_objects = await asyncio.to_thread(
        _list_objects, settings.s3.bucket, prefix="evidence/"
    )

    rows = []
    for obj in objects[:200]:
        key: str = obj["Key"]
        url = _file_url(CLIPS_BUCKET, key)
        size_kb = obj["Size"] // 1024
        ts = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S UTC")
        ext = key.rsplit(".", 1)[-1].lower()
        name = key.split("/")[-1]
        stream = key.split("/")[1] if key.count("/") >= 2 else "-"

        if ext == "mp4":
            preview = (
                '<video controls width="300" height="169" preload="metadata" style="border-radius:4px">' +
                f'<source src="{url}" type="video/mp4"></video>'
            )
            badge = '<span class="badge video">MP4</span>'
        elif ext in ("jpg", "jpeg"):
            preview = (
                f'<img src="{url}" height="120" loading="lazy" ' +
                'style="max-width:200px;object-fit:contain;border-radius:4px">'
            )
            badge = '<span class="badge image">JPEG</span>'
        else:
            preview = "-"
            badge = f'<span class="badge">{ext.upper()}</span>'

        rows.append(f"""<tr>
          <td>{badge}</td>
          <td>{stream}</td>
          <td style="font-size:0.75rem;word-break:break-all">{name}</td>
          <td>{size_kb} KB</td>
          <td>{ts}</td>
          <td>{preview}</td>
          <td><a href="{url}" download="{name}">&#11015;</a></td>
        </tr>""")

    table = "\n".join(rows) if rows else "<tr><td colspan=7>No clips yet</td></tr>"
    json_count = len(json_objects)
    mp4_count = sum(1 for o in objects if o["Key"].endswith(".mp4"))
    jpg_count = sum(1 for o in objects if o["Key"].endswith((".jpg", ".jpeg")))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>InferOps Evidence Browser</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 0; }}
    h1 {{ background: #1a1d2e; padding: 14px 24px; margin: 0; font-size: 1.2rem; display: flex; align-items: center; justify-content: space-between; }}
    .stats {{ display: flex; gap: 12px; padding: 16px 24px; background: #141624; flex-wrap: wrap; }}
    .stat {{ background: #1e2235; border-radius: 8px; padding: 12px 20px; min-width: 130px; }}
    .stat-val {{ font-size: 2rem; font-weight: bold; color: #7c9eff; }}
    .stat-lbl {{ font-size: 0.78rem; color: #888; margin-top: 4px; }}
    .container {{ overflow-x: auto; padding: 0 24px 32px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th {{ background: #1a1d2e; padding: 10px 10px; text-align: left; position: sticky; top: 0; z-index: 1; }}
    td {{ padding: 10px 10px; border-bottom: 1px solid #1e2235; vertical-align: middle; }}
    tr:hover td {{ background: #181b2c; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: bold; }}
    .badge.video {{ background: #2d4a7a; color: #7cb9ff; }}
    .badge.image {{ background: #3a2d1a; color: #ffb85c; }}
    .badge {{ background: #2d2d2d; color: #aaa; }}
    a {{ color: #7c9eff; text-decoration: none; font-size: 1.2rem; }}
    a:hover {{ text-decoration: underline; }}
    .refresh {{ background: #2a3f6f; color: #fff; border: none; padding: 7px 16px; border-radius: 6px; cursor: pointer; font-size: 0.82rem; }}
  </style>
</head>
<body>
  <h1>&#127909; InferOps Evidence Browser
    <button class="refresh" onclick="location.reload()">&#8635; Refresh</button>
  </h1>
  <div class="stats">
    <div class="stat"><div class="stat-val">{len(objects)}</div><div class="stat-lbl">Total Clips</div></div>
    <div class="stat"><div class="stat-val">{mp4_count}</div><div class="stat-lbl">MP4 Videos</div></div>
    <div class="stat"><div class="stat-val">{jpg_count}</div><div class="stat-lbl">JPEG Snapshots</div></div>
    <div class="stat"><div class="stat-val">{json_count}</div><div class="stat-lbl">JSON Records</div></div>
  </div>
  <div class="container">
    <table>
      <thead><tr>
        <th>Type</th><th>Stream</th><th>File</th>
        <th>Size</th><th>Saved at</th><th>Preview</th><th>DL</th>
      </tr></thead>
      <tbody>{table}</tbody>
    </table>
  </div>
</body>
</html>"""
    return HTMLResponse(html)
