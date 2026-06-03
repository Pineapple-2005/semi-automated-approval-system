"""Review router — GET /review/queue and POST /review/{job_id}."""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import VerificationJob
from app.utils.minio_client import get_presigned_url
from app.tasks.webhook_sender import send_webhook

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])


# ---------------------------------------------------------------------------
# Shared dependency  (mirrors verification.py to keep routers self-contained)
# ---------------------------------------------------------------------------


async def get_api_key_system(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """Validate X-API-Key header and return the associated system_id."""
    is_valid, system_id = settings.validate_api_key(x_api_key)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return system_id


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    action: str  # "approve" or "reject"
    reviewed_by: str | None = None
    review_comment: str | None = None


class WebhookPayload(BaseModel):
    job_id: str
    system_id: str
    user_id: str
    status: str
    flags: list
    approval_signals: list
    extracted_data: dict
    confidence_scores: dict
    document_type: str
    document_type_detected: str | None
    reviewed_by: str | None
    review_comment: str | None
    reviewed_at: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_to_dict(job: VerificationJob, presigned_url: str | None = None) -> dict[str, Any]:
    """Serialise a VerificationJob ORM instance to a plain dict.

    JSONB columns are passed through as-is (already Python dicts/lists).
    Datetime columns are converted to ISO-8601 strings.
    """

    def _iso(dt) -> str | None:  # noqa: ANN001
        return dt.isoformat() if dt is not None else None

    return {
        "job_id": str(job.job_id),
        "system_id": job.system_id,
        "user_id": job.user_id,
        "document_type": job.document_type,
        "expected_name": job.expected_name,
        "expected_dob": job.expected_dob,
        "callback_url": job.callback_url,
        "status": job.status,
        "flags": job.flags if job.flags is not None else [],
        "approval_signals": job.approval_signals if job.approval_signals is not None else [],
        "extracted_data": job.extracted_data if job.extracted_data is not None else {},
        "confidence_scores": (
            job.confidence_scores if job.confidence_scores is not None else {}
        ),
        "document_type_detected": job.document_type_detected,
        "image_path": job.image_path,
        "created_at": _iso(job.created_at),
        "reviewed_by": job.reviewed_by,
        "review_comment": job.review_comment,
        "reviewed_at": _iso(job.reviewed_at),
        "presigned_image_url": presigned_url,
    }


def _dispatch_review_webhook(job: VerificationJob) -> None:
    """Build and enqueue the post-review webhook payload (fire-and-forget)."""
    if not job.callback_url:
        return
    payload = WebhookPayload(
        job_id=str(job.job_id),
        system_id=job.system_id,
        user_id=job.user_id,
        status=job.status,
        flags=job.flags or [],
        approval_signals=job.approval_signals or [],
        extracted_data=job.extracted_data or {},
        confidence_scores=job.confidence_scores or {},
        document_type=job.document_type,
        document_type_detected=job.document_type_detected,
        reviewed_by=job.reviewed_by,
        review_comment=job.review_comment,
        reviewed_at=job.reviewed_at.isoformat() if job.reviewed_at else None,
    )
    send_webhook.delay(str(job.job_id), job.callback_url, payload.model_dump())
    logger.info("Webhook dispatched for job %s to %s", job.job_id, job.callback_url)


def _safe_presigned(image_path: str, job_id: str) -> str | None:
    """Return a presigned URL, logging and suppressing any MinIO errors."""
    try:
        return get_presigned_url(image_path)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not generate presigned URL for job %s: %s", job_id, exc
        )
        return None


# ---------------------------------------------------------------------------
# GET /review/queue
# ---------------------------------------------------------------------------


@router.get("/review/queue")
def get_review_queue(
    system_id: str = Query(..., description="Tenant system_id to filter the queue"),
    key_system_id: str = Depends(get_api_key_system),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return all jobs with status ``needs_review`` for the given system."""
    if system_id != key_system_id:
        raise HTTPException(
            status_code=403,
            detail="system_id query param does not match API key ownership",
        )

    jobs: list[VerificationJob] = (
        db.query(VerificationJob)
        .filter(
            VerificationJob.system_id == system_id,
            VerificationJob.status == "needs_review",
        )
        .order_by(VerificationJob.created_at.desc())
        .all()
    )

    items = [
        _job_to_dict(job, _safe_presigned(job.image_path, str(job.job_id)))
        for job in jobs
    ]

    return {"items": items, "total": len(items)}


# ---------------------------------------------------------------------------
# POST /review/{job_id}
# ---------------------------------------------------------------------------


@router.post("/review/{job_id}")
def submit_review(
    job_id: str,
    review: ReviewRequest,
    key_system_id: str = Depends(get_api_key_system),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Apply an ``approved`` or ``rejected`` decision to a ``needs_review`` job."""
    # 1. Validate action value
    if review.action not in ("approve", "reject"):
        raise HTTPException(
            status_code=422,
            detail="action must be 'approve' or 'reject'",
        )

    # 2. Fetch job
    job: VerificationJob | None = (
        db.query(VerificationJob)
        .filter(VerificationJob.job_id == job_id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # 3. Ownership check
    if job.system_id != key_system_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this job",
        )

    # 4. Must be in needs_review state
    if job.status != "needs_review":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job '{job_id}' is in status '{job.status}' and cannot be "
                "reviewed (only 'needs_review' jobs are reviewable)"
            ),
        )

    # 5. Apply the review ("approve" → "approved", "reject" → "rejected")
    job.status = "approved" if review.action == "approve" else "rejected"
    job.reviewed_by = review.reviewed_by or "admin"
    job.review_comment = review.review_comment
    job.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)

    # 6. Fire-and-forget webhook if callback_url is present
    _dispatch_review_webhook(job)

    presigned_url = _safe_presigned(job.image_path, str(job.job_id))
    return _job_to_dict(job, presigned_url)
