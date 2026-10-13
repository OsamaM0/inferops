"""S3/MinIO client (async via thread pool)."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from functools import lru_cache

import boto3
from botocore.client import Config

from inferops.core.config import settings


@lru_cache(maxsize=1)
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3.endpoint,
        aws_access_key_id=settings.s3.access_key,
        aws_secret_access_key=settings.s3.secret_key,
        region_name=settings.s3.region,
        config=Config(signature_version="s3v4"),
    )


@lru_cache(maxsize=1)
def _public_s3_client():
    """Client whose presigned URLs use the public/external endpoint URL."""
    pub = settings.s3.public_url or settings.s3.endpoint
    return boto3.client(
        "s3",
        endpoint_url=pub,
        aws_access_key_id=settings.s3.access_key,
        aws_secret_access_key=settings.s3.secret_key,
        region_name=settings.s3.region,
        config=Config(signature_version="s3v4"),
    )


def presign_url(bucket: str, key: str, expires: int = 3600) -> str:
    """Return a presigned GET URL using the public-facing endpoint."""
    return _public_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def ensure_bucket(bucket: str | None = None) -> str:
    b = bucket or settings.s3.bucket
    c = s3_client()
    try:
        c.head_bucket(Bucket=b)
    except Exception:
        with suppress(Exception):
            c.create_bucket(Bucket=b)
    return b


async def put_bytes(key: str, data: bytes, *, bucket: str | None = None,
                    content_type: str = "application/octet-stream") -> str:
    b = ensure_bucket(bucket)
    await asyncio.to_thread(s3_client().put_object, Bucket=b, Key=key, Body=data, ContentType=content_type)
    return f"s3://{b}/{key}"


async def get_bytes(uri: str) -> bytes | None:
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    b, _, key = rest.partition("/")
    try:
        obj = await asyncio.to_thread(s3_client().get_object, Bucket=b, Key=key)
        return obj["Body"].read()
    except Exception:
        return None
