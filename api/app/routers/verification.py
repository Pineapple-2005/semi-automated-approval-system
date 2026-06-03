"""Verification router — POST /verify and GET /status/{job_id}."""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import VerificationJob
from app.services.system_requirements import ALL_DOCUMENT_TYPES, is_document_type_allowed
from app.tasks.process_document import process_document
from app.utils.minio_client import get_presigned_url, upload_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class VerifyResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    system_id: str
    user_id: str
    document_type: str
    expected_name: str
    expected_dob: str | None
    callback_url: str | None
    status: str
    flags: list
    approval_signals: list
    extracted_data: dict
    confidence_scores: dict
    document_type_detected: str | None
    image_path: str
    created_at: str | None
    reviewed_by: str | None
    review_comment: str | None
    reviewed_at: str | None
    presigned_image_url: str | None


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------


def get_api_key_system(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """Validate X-API-Key header and return the associated system_id."""
    is_valid, system_id = settings.validate_api_key(x_api_key)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return system_id


router = APIRouter(tags=["verification"])


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------


@router.post("/verify", response_model=VerifyResponse, status_code=202)
async def verify_document(
    file: UploadFile,
    user_id: str = Form(...),
    document_type: str = Form(...),
    expected_name: str = Form(...),
    expected_dob: str = Form(None),
    callback_url: str = Form(None),
    system_id_claim: str = Form(..., alias="system_id"),
    system_id: str = Depends(get_api_key_system),
    db: Session = Depends(get_db),
) -> VerifyResponse:
    """Accept a Philippine ID image for asynchronous verification.

    Validates ownership, document type, file type, and file size before
    uploading to MinIO and enqueuing the Celery processing task.
    """
    # 1. system_id claim must match the key-derived system_id
    if system_id_claim != system_id:
        raise HTTPException(
            status_code=403,
            detail="system_id in form does not match API key ownership",
        )

    # 2a. Global document-type allow-list
    if document_type not in ALL_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown document_type '{document_type}'. "
                f"Valid types: {sorted(ALL_DOCUMENT_TYPES)}"
            ),
        )

    # 2b. System-specific document-type restriction
    if not is_document_type_allowed(system_id, document_type):
        raise HTTPException(
            status_code=403,
            detail=(
                f"System '{system_id}' is not permitted to submit "
                f"document_type '{document_type}'."
            ),
        )

    # 3. Content-type must be an image
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file must be an image (got '{content_type}')",
        )

    # 4. Read bytes and enforce size limit
    file_bytes = await file.read()
    max_bytes = int(settings.MAX_IMAGE_SIZE_MB * 1024 * 1024)
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {len(file_bytes)} bytes exceeds the "
                f"{settings.MAX_IMAGE_SIZE_MB} MB limit"
            ),
        )

    # 5. Generate job ID and upload to MinIO
    job_id = uuid4()
    filename = file.filename or "document.jpg"
    object_name = f"{system_id}/{job_id}/{filename}"

    try:
        upload_file(
            object_name=object_name,
            file_data=file_bytes,
            content_type=content_type,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("MinIO upload failed for job %s", job_id)
        raise HTTPException(
            status_code=500, detail="Failed to upload document to storage"
        ) from exc

    # 6. Persist job record
    job = VerificationJob(
        job_id=job_id,
        system_id=system_id,
        user_id=user_id,
        document_type=document_type,
        expected_name=expected_name,
        expected_dob=expected_dob,
        callback_url=callback_url,
        status="processing",
        flags=[],
        extracted_data={},
        confidence_scores={},
        image_path=object_name,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 7. Enqueue Celery task
    process_document.delay(str(job_id))
    logger.info("Verification job %s enqueued for system %s", job_id, system_id)

    return VerifyResponse(
        job_id=str(job_id),
        status="processing",
        message="Document submitted for verification",
    )


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    system_id: str = Depends(get_api_key_system),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Return the current status and extracted data for a verification job."""
    job: VerificationJob | None = (
        db.query(VerificationJob)
        .filter(VerificationJob.job_id == job_id)
        .first()
    )

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.system_id != system_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this job",
        )

    # Generate a short-lived presigned URL for the stored image
    presigned_url: str | None = None
    try:
        presigned_url = get_presigned_url(job.image_path)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not generate presigned URL for job %s: %s", job_id, exc
        )

    def _iso(dt) -> str | None:  # noqa: ANN001
        return dt.isoformat() if dt is not None else None

    return JobStatusResponse(
        job_id=str(job.job_id),
        system_id=job.system_id,
        user_id=job.user_id,
        document_type=job.document_type,
        expected_name=job.expected_name,
        expected_dob=job.expected_dob,
        callback_url=job.callback_url,
        status=job.status,
        flags=job.flags if job.flags is not None else [],
        approval_signals=job.approval_signals if job.approval_signals is not None else [],
        extracted_data=job.extracted_data if job.extracted_data is not None else {},
        confidence_scores=(
            job.confidence_scores if job.confidence_scores is not None else {}
        ),
        document_type_detected=job.document_type_detected,
        image_path=job.image_path,
        created_at=_iso(job.created_at),
        reviewed_by=job.reviewed_by,
        review_comment=job.review_comment,
        reviewed_at=_iso(job.reviewed_at),
        presigned_image_url=presigned_url,
    )
