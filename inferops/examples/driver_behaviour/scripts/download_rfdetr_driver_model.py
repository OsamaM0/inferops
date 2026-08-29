"""Download the trained RF-DETR driver-behaviour ONNX model.

The default target is the repo-local Triton-style model repository. When this
workspace's live Triton repository exists at /workspace/models, the script also
copies the model there so the running server can load it immediately.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

DEFAULT_FILE_ID = "1NmZdondieMuqWeKOkWYPSvxBWdnUI56C"
MODEL_NAME = "rfdetr_driver_behaviour_onnx"
MIN_MODEL_BYTES = 10 * 1024 * 1024


def _domain_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output() -> Path:
    return _domain_root() / "model_repository" / MODEL_NAME / "1" / "model.onnx"


def _default_live_output() -> Path | None:
    env_path = os.getenv("INFEROPS_LIVE_RFDETR_MODEL_PATH")
    if env_path:
        return Path(env_path)
    live_repo = Path("/workspace/models")
    if live_repo.exists():
        return live_repo / MODEL_NAME / "1" / "model.onnx"
    return None


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 inferops-model-downloader/1.0",
        },
    )


def _download_response(opener: urllib.request.OpenerDirector, url: str, output: Path) -> None:
    with opener.open(_request(url), timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            html = response.read().decode("utf-8", errors="replace")
            next_url = _confirmed_drive_url(html)
            if not next_url:
                raise RuntimeError("Google Drive returned HTML without a downloadable confirmation form")
            _download_response(opener, next_url, output)
            return

        with output.open("wb") as file_obj:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)


def _confirmed_drive_url(html: str) -> str | None:
    form = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>', html, re.S)
    if not form:
        return None
    action = form.group(1).replace("&amp;", "&")
    body = form.group(2)
    params = {}
    for name, value in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body):
        params[name] = value.replace("&amp;", "&")
    return f"{action}?{urllib.parse.urlencode(params)}"


def download_drive_file(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    url = f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}"

    fd, tmp_name = tempfile.mkstemp(prefix="model.", suffix=".onnx", dir=output.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _download_response(opener, url, tmp_path)
        _verify_model(tmp_path)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _verify_model(path: Path) -> None:
    size = path.stat().st_size
    head = path.read_bytes()[:64]
    if size < MIN_MODEL_BYTES:
        raise RuntimeError(f"downloaded file is too small to be the trained model: {size} bytes")
    if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        raise RuntimeError("downloaded file is HTML, not an ONNX model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="Google Drive file id")
    parser.add_argument("--output", type=Path, default=_default_output(), help="Primary model.onnx output path")
    parser.add_argument("--live-output", type=Path, default=_default_live_output(), help="Optional live Triton model.onnx path")
    parser.add_argument("--no-live-copy", action="store_true", help="Do not copy to the live Triton model path")
    args = parser.parse_args()

    output = args.output.resolve()
    print(f"Downloading RF-DETR driver model to {output}")
    download_drive_file(args.file_id, output)
    print(f"Wrote {output} ({output.stat().st_size} bytes)")

    live_output = args.live_output.resolve() if args.live_output else None
    if live_output and not args.no_live_copy and live_output != output:
        live_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, live_output)
        print(f"Copied model to live Triton path {live_output} ({live_output.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
