from __future__ import annotations

import asyncio
import hashlib
import time
from urllib.parse import urlparse


class S3Storage:
    def __init__(self, *, endpoint=None, access_key=None, secret_key=None, bucket="inferops", region="us-east-1"):
        self.options = dict(endpoint_url=endpoint, aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key, region_name=region)
        self.bucket = bucket
        self.client = None
        self._start_lock = asyncio.Lock()

    async def start(self):
        async with self._start_lock:
            if self.client is not None:
                return
            import boto3
            from botocore.config import Config
            client = await asyncio.to_thread(boto3.client, "s3", **self.options,
                                            config=Config(signature_version="s3v4", connect_timeout=5,
                                                          read_timeout=15, retries={"max_attempts": 2}))
            try:
                try:
                    await asyncio.to_thread(client.head_bucket, Bucket=self.bucket)
                except client.exceptions.ClientError as exc:
                    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    if status != 404:
                        raise
                    options = {"Bucket": self.bucket}
                    if self.options["region_name"] != "us-east-1":
                        options["CreateBucketConfiguration"] = {"LocationConstraint": self.options["region_name"]}
                    await asyncio.to_thread(client.create_bucket, **options)
            except BaseException:
                await asyncio.to_thread(client.close)
                raise
            self.client = client

    async def close(self):
        if self.client is not None:
            await asyncio.to_thread(self.client.close)
            self.client = None

    def _location(self, uri):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket or not parsed.path.lstrip("/"):
            raise ValueError("artifact is outside configured object store")
        return parsed.path.lstrip("/")

    async def put(self, key, data, content_type="application/octet-stream"):
        from inferops.adapters.outbound.observability.metrics import EVIDENCE_UPLOAD_LATENCY, EVIDENCE_UPLOADS
        kind = key.rsplit("/", 1)[-1].split(".", 1)[0]
        started = time.monotonic()
        try:
            uri = await self._put(key, data, content_type)
        except Exception:
            EVIDENCE_UPLOADS.labels(bucket=self.bucket, kind=kind, result="error").inc()
            raise
        else:
            EVIDENCE_UPLOADS.labels(bucket=self.bucket, kind=kind, result="ok").inc()
            return uri
        finally:
            EVIDENCE_UPLOAD_LATENCY.labels(bucket=self.bucket, kind=kind).observe(time.monotonic() - started)

    async def _put(self, key, data, content_type):
        await self.start()
        digest = hashlib.sha256(data).hexdigest()
        uri = f"s3://{self.bucket}/{key}"
        if not await self.verify(uri, digest):
            await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=key, Body=data,
                                    ContentType=content_type, Metadata={"sha256": digest})
        if not await self.verify(uri, digest):
            raise OSError("uploaded artifact could not be verified")
        return uri

    async def read(self, uri):
        await self.start()
        response = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=self._location(uri))
        body = response["Body"]
        try:
            return await asyncio.to_thread(body.read)
        finally:
            body.close()

    async def verify(self, uri, sha256=None):
        if self.client is None:
            await self.start()
        try:
            result = await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=self._location(uri))
        except self.client.exceptions.ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise
        if sha256 is None:
            return True
        if result.get("Metadata", {}).get("sha256") != sha256:
            return False
        # Metadata describes the intended bytes. Verify the stored bytes too;
        # ETags are not content checksums for multipart uploads.
        return hashlib.sha256(await self.read(uri)).hexdigest() == sha256

    async def delete(self, uri):
        await self.start()
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=self._location(uri))
