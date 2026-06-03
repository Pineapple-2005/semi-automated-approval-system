"""MinIO client utilities for document storage."""

import io
import json
import logging
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton
_client = None


def get_minio_client() -> Minio:
    """Create and return the MinIO singleton client.

    Uses MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, and
    MINIO_SECURE from application settings.
    """
    global _client
    if _client is None:
        _client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        logger.debug("MinIO client initialised (endpoint=%s)", settings.MINIO_ENDPOINT)
    return _client


def ensure_bucket_exists(bucket_name: str) -> None:
    """Create *bucket_name* if it does not already exist and apply a
    public-read-like bucket policy so pre-signed URLs work correctly.

    Args:
        bucket_name: Name of the target MinIO bucket.

    Raises:
        S3Error: If the bucket cannot be created or the policy cannot be set.
    """
    client = get_minio_client()
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info("Created MinIO bucket '%s'", bucket_name)

        # Public-read policy (s3:GetObject for every principal)
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        client.set_bucket_policy(bucket_name, json.dumps(policy))
        logger.debug("Public-read policy applied to bucket '%s'", bucket_name)
    except S3Error as exc:
        logger.error(
            "Failed to ensure bucket '%s' exists: %s", bucket_name, exc
        )
        raise


def upload_file(
    object_name: str,
    file_data: bytes,
    content_type: str = "image/jpeg",
) -> str:
    """Upload *file_data* to the configured MinIO bucket.

    Args:
        object_name:  Destination object key inside the bucket.
        file_data:    Raw bytes to upload.
        content_type: MIME type header sent with the object (default
                      ``"image/jpeg"``).

    Returns:
        The *object_name* that was used for the upload.

    Raises:
        S3Error: If the upload fails.
    """
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET
    try:
        ensure_bucket_exists(bucket)
        data_stream = io.BytesIO(file_data)
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=data_stream,
            length=len(file_data),
            content_type=content_type,
        )
        logger.info(
            "Uploaded object '%s' (%d bytes) to bucket '%s'",
            object_name,
            len(file_data),
            bucket,
        )
        return object_name
    except S3Error as exc:
        logger.error(
            "Failed to upload object '%s' to bucket '%s': %s",
            object_name,
            bucket,
            exc,
        )
        raise


def get_presigned_url(object_name: str, expires_hours: int = 24) -> str:
    """Generate a presigned GET URL for *object_name*.

    Args:
        object_name:  Object key inside the configured bucket.
        expires_hours: Validity window in hours (default 24).

    Returns:
        A presigned URL string that allows unauthenticated GET access for
        the specified duration.

    Raises:
        S3Error: If the presigned URL cannot be generated.
    """
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET
    try:
        url = client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(hours=expires_hours),
        )
        logger.debug(
            "Generated presigned URL for object '%s' (expires in %dh)",
            object_name,
            expires_hours,
        )
        return url
    except S3Error as exc:
        logger.error(
            "Failed to generate presigned URL for object '%s': %s",
            object_name,
            exc,
        )
        raise


def download_file(object_name: str) -> bytes:
    """Download an object from the configured MinIO bucket and return its
    raw bytes.

    Args:
        object_name: Object key inside the configured bucket.

    Returns:
        The raw bytes of the downloaded object.

    Raises:
        S3Error: If the download fails.
    """
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET
    response = None
    try:
        response = client.get_object(
            bucket_name=bucket,
            object_name=object_name,
        )
        data = response.read()
        logger.info(
            "Downloaded object '%s' (%d bytes) from bucket '%s'",
            object_name,
            len(data),
            bucket,
        )
        return data
    except S3Error as exc:
        logger.error(
            "Failed to download object '%s' from bucket '%s': %s",
            object_name,
            bucket,
            exc,
        )
        raise
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def delete_file(object_name: str) -> None:
    """Delete an object from the configured MinIO bucket.

    Args:
        object_name: Object key inside the configured bucket to delete.

    Raises:
        S3Error: If the deletion fails.
    """
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET
    try:
        client.remove_object(
            bucket_name=bucket,
            object_name=object_name,
        )
        logger.info(
            "Deleted object '%s' from bucket '%s'", object_name, bucket
        )
    except S3Error as exc:
        logger.error(
            "Failed to delete object '%s' from bucket '%s': %s",
            object_name,
            bucket,
            exc,
        )
        raise
