from __future__ import annotations

import logging
from io import BytesIO

import boto3
from botocore.config import Config as BotoConfig

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_client() -> "boto3.client":  # type: ignore[name-defined]
    s = get_settings()
    scheme = "https" if s.minio_use_ssl else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{s.minio_endpoint}",
        aws_access_key_id=s.minio_access_key,
        aws_secret_access_key=s.minio_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    settings = get_settings()
    client = _get_client()
    client.put_object(
        Bucket=settings.minio_bucket,
        Key=key,
        Body=BytesIO(data),
        ContentType=content_type,
    )
    logger.debug("Uploaded %s (%d bytes)", key, len(data))


def download_bytes(key: str) -> bytes:
    settings = get_settings()
    client = _get_client()
    resp = client.get_object(Bucket=settings.minio_bucket, Key=key)
    return resp["Body"].read()  # type: ignore[no-any-return]


def presigned_url(key: str, expires_in: int = 3600) -> str:
    settings = get_settings()
    client = _get_client()
    return client.generate_presigned_url(  # type: ignore[no-any-return]
        "get_object",
        Params={"Bucket": settings.minio_bucket, "Key": key},
        ExpiresIn=expires_in,
    )
