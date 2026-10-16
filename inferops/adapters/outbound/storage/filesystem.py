from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


class FilesystemStorage:
    def __init__(self, root):
        self.root = Path(root).resolve()

    async def start(self):
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def close(self):
        pass

    def _path(self, key):
        if key.startswith("file://"):
            path = Path(unquote(urlparse(key).path)).resolve()
        else:
            path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("object key escapes storage root")
        return path

    async def put(self, key, data, content_type="application/octet-stream"):
        path = self._path(key)
        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        await asyncio.to_thread(write)
        return path.as_uri()

    async def read(self, uri):
        return await asyncio.to_thread(self._path(uri).read_bytes)

    async def verify(self, uri, sha256=None):
        path = self._path(uri)
        if not path.is_file():
            return False
        if sha256 is None:
            return True
        return hashlib.sha256(await self.read(uri)).hexdigest() == sha256

    async def delete(self, uri):
        await asyncio.to_thread(self._path(uri).unlink, missing_ok=True)
